"""End-to-end orchestration of video I/O, tracking, ReID and persistence.

The collaborators in this module perform the specialized work; the
orchestrator owns their order, run-local state and lifecycle.  Start with
``PersonReIdPipeline.process`` when tracing a complete analysis.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from app.config import AppPaths, PipelineConfig
from app.evaluation.artifacts import RunArtifacts
from app.pipeline.contracts import BackendMetadata, EmbeddingEncoder, PersonTracker, ReleasableBackend, TrackerFactory
from app.pipeline.detector_tracker import build_tracker
from app.pipeline.detail_tracking import (
    DetailTrackingPolicy,
    bbox_center,
    has_sufficient_new_person_evidence,
    intersection_over_smaller_box,
)
from app.pipeline.reid_encoder import build_encoder
from app.reid.repository import EmbeddingBatch, EvaluationRepository, ProfileManager, ProfileMatcher, ProfileUpdater
from app.reid.service import ProfileService
from app.reid.embeddings import checked_embedding
from app.storage.encoder_paths import paths_for_encoder
from app.storage.models import PipelineResult
from app.storage.vector_store import SQLiteVectorStore
from app.utils.camera_utils import CameraSource, open_camera_capture
from app.utils.id_utils import make_run_id, safe_source_name
from app.utils.image_utils import (
    crop_quality_score,
    crop_xyxy,
    draw_detection,
    is_valid_crop,
    normalize_vector,
    person_bbox_overlap_ratio,
    save_crop,
)
from app.utils.detail_utils import DetailSnapshot, compact_detail_influences, extract_detail_snapshot

ProgressCallback = Callable[[int, int | None, str], None]
FrameCallback = Callable[[int, np.ndarray], None]


class Releasable(Protocol):
    def release(self) -> None:
        ...


@dataclass
class TrackEmbeddingCandidate:
    """One quality-approved crop and embedding buffered for a tracker ID."""

    embedding: np.ndarray
    quality_score: float
    quality_details: dict[str, float]
    crop: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    frame_index: int
    detection_confidence: float
    detail_snapshot: DetailSnapshot | None = None


class PersonReIdPipeline:
    """Coordinate one configured person re-identification pipeline.

    Construction prepares persistence; heavyweight backends are loaded inside
    :meth:`process` so startup failures receive a manifest. It handles one source and returns its artifacts
    and counters as a ``PipelineResult``.
    """

    def __init__(self, config: PipelineConfig, paths: AppPaths | None = None, *,
                 tracker: PersonTracker | None = None, encoder: EmbeddingEncoder | None = None,
                 store: EvaluationRepository | None = None,
                 profiles: ProfileManager | None = None,
                 matcher: ProfileMatcher | None = None, updater: ProfileUpdater | None = None,
                 detail_policy: DetailTrackingPolicy | None = None,
                 tracker_factory: TrackerFactory = build_tracker) -> None:
        if profiles is not None and (matcher is not None or updater is not None):
            raise ValueError("Configure matcher/updater on the supplied profile manager, not on the pipeline.")
        self.config = config
        base_paths = paths if paths is not None else AppPaths()
        # Base paths do not opt out of encoder isolation. Explicit repositories
        # own their namespace and must not be rewritten by default composition.
        self.paths = (paths_for_encoder(base_paths, config)
                      if store is None and profiles is None else base_paths)
        self.paths.ensure()
        self.store = store if store is not None else SQLiteVectorStore(self.paths.db_path)
        self.profiles = profiles if profiles is not None else ProfileService(
            self.store, matcher=matcher, updater=updater, min_update_similarity=config.min_update_similarity)
        # Lazy loading makes model-start failures visible in the run manifest.
        # Injected collaborators enable integration tests without model downloads.
        self.tracker = tracker
        self.encoder = encoder
        self.tracker_factory = tracker_factory
        self.detail_policy = (
            detail_policy if detail_policy is not None
            else DetailTrackingPolicy.from_config(config)
            if config.decision_policy == "details_tracking_v2"
            else None
        )
        self._has_processed = False

    def _open_capture(self, source: str | int | CameraSource) -> cv2.VideoCapture:
        if isinstance(source, CameraSource):
            return open_camera_capture(source)
        if isinstance(source, int) and sys.platform.startswith("win"):
            return cv2.VideoCapture(source, cv2.CAP_DSHOW)
        return cv2.VideoCapture(source)

    @staticmethod
    def _source_to_label(source: str | int | CameraSource) -> str:
        if isinstance(source, CameraSource):
            return source.label()
        return str(source)

    @staticmethod
    def _combined_embedding(candidates: list[TrackEmbeddingCandidate]) -> np.ndarray:
        return checked_embedding(PersonReIdPipeline._embedding_batch(candidates).embedding_sum).astype(np.float32)

    @staticmethod
    def _embedding_batch(candidates: list[TrackEmbeddingCandidate]) -> EmbeddingBatch:
        if not candidates:
            raise ValueError("Cannot build an embedding from an empty candidate list.")
        embeddings = np.stack([checked_embedding(candidate.embedding) for candidate in candidates], axis=0)
        weights = np.asarray([max(candidate.quality_score, 0.05) for candidate in candidates], dtype=np.float64)
        return EmbeddingBatch(np.sum(embeddings * weights[:, None], axis=0), float(weights.sum()), len(candidates))

    @staticmethod
    def _best_candidate(candidates: list[TrackEmbeddingCandidate]) -> TrackEmbeddingCandidate:
        if not candidates:
            raise ValueError("Cannot select a best candidate from an empty candidate list.")
        return max(candidates, key=lambda candidate: candidate.quality_score)

    def _quality_payload(
        self,
        *,
        run_id: str,
        event_type: str,
        candidate: TrackEmbeddingCandidate,
        good_frame_count: int,
        quality_average: float | None = None,
        decision_frame_index: int,
        detail_vector: np.ndarray | list[float] | tuple[float, ...] | None = None,
        match: object | None = None,
    ) -> dict[str, object]:
        quality_average = candidate.quality_score if quality_average is None else quality_average
        payload: dict[str, object] = {
            "run_id": run_id,
            "mode_id": self.config.mode_id,
            "pipeline_type": self.config.pipeline_type,
            "event_type": event_type,
            "decision_frame_index": decision_frame_index,
            "snapshot_frame_index": candidate.frame_index,
            "det_confidence": float(candidate.detection_confidence),
            "quality_score": float(candidate.quality_score),
            "quality_average": float(quality_average),
            "good_frame_count": int(good_frame_count),
            "embedding_weight": float(candidate.quality_score),
            "snapshot_quality": float(candidate.quality_score),
            "quality_details": {key: float(value) for key, value in candidate.quality_details.items()},
        }
        if candidate.detail_snapshot is not None:
            payload["details"] = candidate.detail_snapshot.to_payload()
            payload["detail_vector"] = (
                [float(value) for value in np.asarray(detail_vector).reshape(-1)]
                if detail_vector is not None else list(candidate.detail_snapshot.vector)
            )
            payload["detail_weight"] = float(quality_average)
            payload["detail_reliability"] = float(candidate.detail_snapshot.reliability)
        if match is not None:
            breakdown = getattr(match, "detail_breakdown", None)
            payload["match_explanation"] = {
                "person_id": getattr(match, "person_id", None),
                "final_score": float(getattr(match, "score", 0.0)),
                "visual_score": getattr(match, "visual_score", None),
                "detail_score": getattr(match, "detail_score", None),
                "detail_weight": float(getattr(match, "detail_weight", 0.0)),
                "motion_bonus": float(getattr(match, "motion_bonus", 0.0)),
                "decision_zone": getattr(match, "decision_zone", "unknown"),
                "summary": compact_detail_influences(breakdown),
                "top_matches": getattr(match, "top_matches", []),
            }
        return payload

    def process(
        self,
        source: str | int | CameraSource,
        progress_callback: ProgressCallback | None = None,
        frame_callback: FrameCallback | None = None,
        max_duration_seconds: float | None = None,
    ) -> PipelineResult:
        """Analyze a video or camera source and persist identity observations.

        ``progress_callback`` receives coarse run progress. ``frame_callback``
        receives annotated preview frames and may be used by a UI, but must not
        mutate the frame.  Identity and candidate maps are local to this call;
        durable person profiles and analysis events live in the vector store.

        Raises:
            RuntimeError: If the source cannot be opened.
        """

        if max_duration_seconds is not None:
            max_duration_seconds = float(max_duration_seconds)
            if not np.isfinite(max_duration_seconds) or max_duration_seconds <= 0:
                raise ValueError("max_duration_seconds must be finite and greater than zero.")

        # Several narrow integration tests construct the orchestrator without
        # calling __init__; missing optional policy state must mean main mode.
        detail_policy = getattr(self, "detail_policy", None)

        if getattr(self, "_has_processed", False):
            raise RuntimeError("A pipeline instance handles one source only; construct a fresh tracker/pipeline for the next run.")
        self._has_processed = True
        total_started = time.perf_counter()
        run_id = make_run_id()
        artifacts = RunArtifacts(run_id=run_id, config=self.config, paths=self.paths,
                                 source=self._source_to_label(source))
        self.last_manifest_path = artifacts.manifest_path
        resources: list[Releasable] = [artifacts]
        error: BaseException | None = None
        result: PipelineResult | None = None
        load_seconds = 0.0
        processing_started: float | None = None
        try:
            run_metadata = asdict(self.config)
            artifacts.metadata["identity_decision_policy"] = (
                detail_policy.as_metadata() if detail_policy is not None
                else {"name": "main_single_threshold", "match_threshold": self.config.match_threshold}
            )
            if detail_policy is not None:
                run_metadata["identity_decision_policy"] = detail_policy.as_metadata()
            if max_duration_seconds is not None:
                run_metadata["max_duration_seconds"] = max_duration_seconds
            self.store.add_analysis_run(
                run_id=run_id, mode_id=self.config.mode_id, mode_name=self.config.mode_name,
                pipeline_type=self.config.pipeline_type, source=self._source_to_label(source),
                fps=None, frame_count=None, width=None, height=None, metadata=run_metadata,
            )
            load_started = time.perf_counter()
            try:
                if getattr(self, "tracker", None) is None:
                    self.tracker = self.tracker_factory(self.config)
                if isinstance(self.tracker, ReleasableBackend):
                    resources.append(self.tracker)
                if getattr(self, "encoder", None) is None:
                    self.encoder = build_encoder(
                        self.config.encoder_backend, device=self.config.device,
                        model_name=self.config.reid_model_name, checkpoint_path=self.config.reid_checkpoint,
                    )
                if isinstance(self.encoder, ReleasableBackend) and self.encoder is not self.tracker:
                    resources.append(self.encoder)
            finally:
                load_seconds = time.perf_counter() - load_started
            components: dict[str, dict[str, Any]] = {}
            for label, backend in (("tracker", self.tracker), ("encoder", self.encoder), ("profiles", self.profiles)):
                description: dict[str, Any] = {"implementation": f"{type(backend).__module__}.{type(backend).__qualname__}"}
                if isinstance(backend, BackendMetadata):
                    description["metadata"] = backend.describe_backend()
                components[label] = description
            artifacts.metadata["components"] = components
            # Configured defaults must not masquerade as actual injected models.
            artifacts.metadata["configured_models"] = artifacts.metadata["models"]
            tracker_metadata = components["tracker"].get("metadata", {})
            encoder_metadata = components["encoder"].get("metadata", {})
            unknown_model = {"sha256": None, "note": "Injected backend supplied no model reference."}
            artifacts.metadata["models"] = {
                "detector": tracker_metadata.get("detector", dict(unknown_model)),
                "tracker": tracker_metadata.get("tracker", dict(unknown_model)),
                "encoder": encoder_metadata.get("encoder", {
                    "backend": components["encoder"]["implementation"], "model_name": None,
                    "checkpoint": None, "note": "Injected backend supplied no model reference.",
                }),
            }
            artifacts.metadata["environment"]["requested_device"] = self.config.device
            artifacts.metadata["environment"]["encoder_device"] = getattr(self.encoder, "device", "cpu")
            try:
                import torch
                artifacts.metadata["environment"]["compute"] = {
                    "cpu_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
                    "cuda_runtime": torch.version.cuda,
                    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                }
            except ImportError:
                artifacts.metadata["environment"]["compute"] = {"cuda_available": None}
            processing_started = time.perf_counter()
            cap = self._open_capture(source)
            resources.append(cap)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video source: {source}")
            result = self._process_open_capture(
                source,
                cap,
                resources,
                artifacts,
                progress_callback=progress_callback,
                frame_callback=frame_callback,
                max_duration_seconds=max_duration_seconds,
                run_metadata=run_metadata,
            )
            return result
        except BaseException as exc:
            error = exc
            raise
        finally:
            release_error = None
            for resource in reversed(resources):
                try:
                    resource.release()
                except Exception as exc:
                    release_error = release_error or exc
            final_error = error or release_error
            status = "failed" if final_error else "completed"
            processing_seconds = time.perf_counter() - processing_started if processing_started else 0.0
            try:
                artifacts.finish(status=status, error=final_error, processing_seconds=processing_seconds,
                                 model_load_seconds=load_seconds, total_seconds=time.perf_counter() - total_started)
                self.store.finish_analysis_run(run_id, status=status, processed_frames=artifacts.processed_frames,
                                               error=str(final_error) if final_error else None)
                artifacts.record_database_after(self.paths.db_path)
            except Exception as exc:
                # A DB finalization failure must not leave a success manifest.
                if error is None:
                    try:
                        artifacts.finish(status="failed", error=exc, processing_seconds=processing_seconds,
                                         model_load_seconds=load_seconds,
                                         total_seconds=time.perf_counter() - total_started)
                    except Exception as artifact_error:
                        exc.add_note(f"Could not mark the run manifest failed: {artifact_error}")
                if error is not None:
                    error.add_note(f"Run finalization also failed: {exc}")
                else:
                    raise
            if release_error is not None and error is None:
                raise release_error

    def _process_open_capture(
        self,
        source: str | int | CameraSource,
        cap: cv2.VideoCapture,
        resources: list[Releasable],
        artifacts: RunArtifacts,
        progress_callback: ProgressCallback | None = None,
        frame_callback: FrameCallback | None = None,
        max_duration_seconds: float | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> PipelineResult:
        detail_policy = getattr(self, "detail_policy", None)
        assert self.tracker is not None and self.encoder is not None
        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = total_frames_raw if total_frames_raw > 0 else None
        if max_duration_seconds is not None:
            # A live duration is based on elapsed wall-clock time. It takes
            # precedence over the frame limit, whose duration depends on how
            # quickly inference can consume webcam frames.
            total_for_progress = None
        elif self.config.max_frames > 0:
            total_for_progress = min(total_frames or self.config.max_frames, self.config.max_frames)
        else:
            total_for_progress = total_frames

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        run_id = artifacts.metadata["run_id"]
        artifacts.set_video(
            fps=float(fps), frame_count=total_frames, width=width, height=height,
            fps_fallback=not cap.get(cv2.CAP_PROP_FPS) or cap.get(cv2.CAP_PROP_FPS) <= 1,
            capture_duration_limit_seconds=max_duration_seconds,
        )
        source_label = safe_source_name(self._source_to_label(source))
        mode_label = safe_source_name(self.config.mode_id)
        output_path = self.paths.output_dir / f"{mode_label}_{run_id}_{source_label}.mp4"
        artifacts.metadata["exports"]["annotated_video"] = str(output_path)
        artifacts.metadata["exports"]["annotated_video_relative_to_artifact_root"] = artifacts.relative_path(output_path)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        resources.append(writer)
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video writer: {output_path}")

        self.store.add_analysis_run(
            run_id=run_id,
            mode_id=self.config.mode_id,
            mode_name=self.config.mode_name,
            pipeline_type=self.config.pipeline_type,
            source=self._source_to_label(source),
            fps=float(fps),
            frame_count=total_frames,
            width=width,
            height=height,
            metadata=run_metadata if run_metadata is not None else asdict(self.config),
        )

        # Run-local bridge from the tracker's short-lived IDs to durable person
        # IDs. Candidates delay the initial decision until enough good views exist.
        track_to_person: dict[int, str] = {}
        track_to_last_score: dict[int, float | None] = {}
        track_candidates: dict[int, list[TrackEmbeddingCandidate]] = {}
        track_low_match_evidence: dict[int, list[tuple[int, float]]] = {}
        track_is_tentative: dict[int, bool] = {}
        recent_person_positions: dict[str, tuple[int, tuple[float, float]]] = {}
        track_candidate_last_frame: dict[int, int] = {}
        track_last_seen_frame: dict[int, int] = {}
        overlap_cooldown_until: dict[int, int] = {}
        created_persons = 0
        matched_events = 0
        strong_match_events = 0
        pending_weak_match_events = 0
        pending_new_person_events = 0
        frame_index = 0
        skipped_low_quality = 0
        skipped_overlap = 0
        expired_track_states = 0
        waiting_for_good_frames = 0
        warnings: list[str] = []

        # Hot path: read -> track -> quality gate -> encode/match -> draw/write.
        capture_started = time.perf_counter()
        while True:
            if max_duration_seconds is not None:
                if frame_index > 0 and time.perf_counter() - capture_started >= max_duration_seconds:
                    break
            elif self.config.max_frames > 0 and frame_index >= self.config.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1

            detections = self.tracker.track_frame(frame)
            active_track_ids = {
                detection.track_id for detection in detections if detection.track_id is not None
            }
            expired_track_ids = [
                track_id for track_id, last_seen in track_last_seen_frame.items()
                if frame_index - last_seen > self.config.track_state_ttl_frames
            ]
            for track_id in expired_track_ids:
                track_to_person.pop(track_id, None)
                track_to_last_score.pop(track_id, None)
                track_candidates.pop(track_id, None)
                track_low_match_evidence.pop(track_id, None)
                track_is_tentative.pop(track_id, None)
                track_candidate_last_frame.pop(track_id, None)
                track_last_seen_frame.pop(track_id, None)
                overlap_cooldown_until.pop(track_id, None)
                expired_track_states += 1
            for track_id in active_track_ids:
                track_last_seen_frame[track_id] = frame_index

            overlap_by_detection: dict[int, float] = {index: 0.0 for index in range(len(detections))}
            for first_index, first in enumerate(detections):
                for second_index in range(first_index + 1, len(detections)):
                    second = detections[second_index]
                    if first.track_id is not None and first.track_id == second.track_id:
                        continue
                    overlap = person_bbox_overlap_ratio(first.bbox_xyxy, second.bbox_xyxy)
                    overlap_by_detection[first_index] = max(overlap_by_detection[first_index], overlap)
                    overlap_by_detection[second_index] = max(overlap_by_detection[second_index], overlap)
                    if overlap > self.config.max_person_overlap_ratio:
                        cooldown_until = frame_index + self.config.overlap_cooldown_frames
                        for track_id in (first.track_id, second.track_id):
                            if track_id is not None:
                                overlap_cooldown_until[track_id] = max(
                                    overlap_cooldown_until.get(track_id, 0), cooldown_until
                                )
            frame_predictions: list[dict[str, object]] = []
            # Keep crops independent of boxes and labels drawn for other people.
            display_frame = frame.copy() if self.config.draw_debug else frame
            assigned_person_boxes_this_frame: list[tuple[str, tuple[int, int, int, int]]] = []
            # Reserve all visible known IDs before visiting unknown tracks.
            used_person_ids_this_frame = {
                track_to_person[detection.track_id]
                for detection in detections
                if detection.track_id in track_to_person
            }

            for detection_index, detection in enumerate(detections):
                person_id = track_to_person.get(detection.track_id)
                score = track_to_last_score.get(detection.track_id)
                person_overlap_ratio = overlap_by_detection[detection_index]
                prediction: dict[str, object] = {
                    "bbox_xyxy": list(detection.bbox_xyxy), "confidence": float(detection.confidence),
                    "class_id": detection.class_id, "track_id": detection.track_id, "person_id": person_id,
                    "match_score": score, "state": "known_track" if person_id else "pending",
                    "quality_score": None, "decision_frame_index": None, "snapshot_frame_index": None,
                    "profile_update_accepted": None, "update_similarity": None, "profile_update_reason": None,
                    "decision_zone": None, "match_visual_score": None, "match_detail_score": None,
                    "match_detail_weight": 0.0, "match_motion_bonus": 0.0,
                    "detail_label": None, "detail_reliability": None,
                    "person_overlap_ratio": person_overlap_ratio,
                    "overlap_cooldown_until_frame": overlap_cooldown_until.get(detection.track_id),
                    "overlap_cooldown_remaining": max(
                        0, overlap_cooldown_until.get(detection.track_id, frame_index) - frame_index
                    ),
                    "quality_details": None,
                    "initial_candidate_count": (
                        len(track_candidates.get(detection.track_id, []))
                        if detection.track_id is not None and person_id is None else None
                    ),
                    "initial_candidate_required": (
                        self.config.min_good_frames_before_reid
                        if detection.track_id is not None and person_id is None else None
                    ),
                    "best_match_person_id": None,
                    "best_match_score": None,
                    "match_threshold": None,
                    "match_reason": None,
                    "eligible_profile_count": None,
                    "excluded_person_ids": None,
                }
                frame_predictions.append(prediction)
                if detection.track_id is None:
                    prediction["state"] = "untracked"
                    if self.config.draw_debug:
                        draw_detection(display_frame, detection, None, None)
                    continue
                currently_overlapping = person_overlap_ratio > self.config.max_person_overlap_ratio
                in_overlap_cooldown = frame_index <= overlap_cooldown_until.get(detection.track_id, 0)
                if person_id is None and currently_overlapping:
                    # Do not combine pre-crossing and post-crossing crops if
                    # the tracker changes identity during an overlap.
                    track_candidates.pop(detection.track_id, None)
                    track_candidate_last_frame.pop(detection.track_id, None)
                    prediction["initial_candidate_count"] = 0

                if person_id is None:
                    last_candidate_frame = track_candidate_last_frame.get(detection.track_id)
                    candidate_due = (
                        last_candidate_frame is None
                        or frame_index - last_candidate_frame >= self.config.initial_candidate_every_n_frames
                    )
                    if not candidate_due and not currently_overlapping and not in_overlap_cooldown:
                        prediction["state"] = "waiting_for_candidate_interval"
                        if self.config.draw_debug:
                            draw_detection(display_frame, detection, person_id, score)
                        continue
                    should_reid = True
                else:
                    should_reid = frame_index % self.config.reid_every_n_frames == 0

                if should_reid:
                    if currently_overlapping or in_overlap_cooldown:
                        prediction["state"] = "overlapping_person" if currently_overlapping else "overlap_cooldown"
                        skipped_overlap += 1
                        if self.config.draw_debug:
                            draw_detection(display_frame, detection, person_id, score)
                        continue

                    crop = crop_xyxy(frame, detection.bbox_xyxy, padding=self.config.crop_padding)
                    if not is_valid_crop(crop, self.config.min_crop_width, self.config.min_crop_height):
                        prediction["state"] = "invalid_crop"
                        skipped_low_quality += 1
                        if self.config.draw_debug:
                            draw_detection(
                                display_frame,
                                detection,
                                person_id,
                                score,
                            )
                        continue

                    quality_score, quality_details = crop_quality_score(
                        crop=crop,
                        bbox_xyxy=detection.bbox_xyxy,
                        frame_shape=frame.shape,
                        detection_confidence=detection.confidence,
                        min_width=self.config.min_crop_width,
                        min_height=self.config.min_crop_height,
                    )
                    prediction["quality_score"] = quality_score
                    prediction["quality_details"] = {
                        key: float(value) for key, value in quality_details.items()
                    }
                    # Unknown tracks use explicit sharpness gates so large or bright
                    # crops cannot hide motion blur in the weighted total score.
                    quality_rejection_state = None
                    if quality_score < self.config.min_embedding_quality:
                        quality_rejection_state = "below_candidate_quality"
                    elif (
                        person_id is None
                        and quality_details.get("blur", 1.0) < self.config.min_initial_blur_score
                    ):
                        quality_rejection_state = "below_initial_blur"
                    elif (
                        person_id is None
                        and quality_details.get("edge_cutoff", 1.0) < 1.0
                        and quality_details.get("blur", 1.0) < self.config.min_border_blur_score
                    ):
                        quality_rejection_state = "below_border_blur"
                    elif person_id is not None and quality_score < self.config.min_update_quality:
                        quality_rejection_state = "below_update_quality"
                    if quality_rejection_state is not None:
                        prediction["state"] = quality_rejection_state
                        skipped_low_quality += 1
                        if self.config.draw_debug:
                            draw_detection(
                                display_frame,
                                detection,
                                person_id,
                                score,
                            )
                        continue

                    embedding = checked_embedding(self.encoder.encode(crop))
                    if hasattr(self, "_embedding_dimension") and embedding.size != self._embedding_dimension:
                        raise ValueError("The encoder changed its embedding dimension within a run.")
                    self._embedding_dimension = embedding.size
                    detail_snapshot = (
                        extract_detail_snapshot(crop, min_confidence=detail_policy.detail_min_confidence)
                        if detail_policy is not None and detail_policy.detail_reranking_enabled else None
                    )
                    candidate = TrackEmbeddingCandidate(
                        embedding=embedding,
                        quality_score=quality_score,
                        quality_details=quality_details,
                        crop=crop.copy(),
                        bbox_xyxy=detection.bbox_xyxy,
                        frame_index=frame_index,
                        detection_confidence=detection.confidence,
                        detail_snapshot=detail_snapshot,
                    )
                    if detail_snapshot is not None:
                        prediction.update(detail_label=detail_snapshot.label,
                                          detail_reliability=detail_snapshot.reliability)

                    # Unknown tracks collect several views before their first
                    # database search, reducing decisions based on one weak crop.
                    if person_id is None:
                        candidates = track_candidates.setdefault(detection.track_id, [])
                        candidates.append(candidate)
                        track_candidate_last_frame[detection.track_id] = frame_index
                        max_buffer_size = max(self.config.min_good_frames_before_reid * 2, 5)
                        if len(candidates) > max_buffer_size:
                            del candidates[0 : len(candidates) - max_buffer_size]
                        prediction["initial_candidate_count"] = len(candidates)

                        if len(candidates) < self.config.min_good_frames_before_reid:
                            prediction["state"] = "waiting_for_initial_observations"
                            waiting_for_good_frames += 1
                            if self.config.draw_debug:
                                draw_detection(
                                    display_frame,
                                    detection,
                                    person_id,
                                    score,
                                )
                            continue

                        combined_embedding = self._combined_embedding(candidates)
                        best_candidate = self._best_candidate(candidates)
                        quality_average = float(np.mean([item.quality_score for item in candidates]))
                        detail_items = [item for item in candidates if item.detail_snapshot is not None]
                        combined_detail_vector = None
                        if detail_items:
                            detail_matrix = np.asarray([item.detail_snapshot.vector for item in detail_items], dtype=np.float32)
                            detail_weights = np.asarray([max(item.quality_score, .05) for item in detail_items], dtype=np.float32)
                            combined_detail_vector = np.average(detail_matrix, axis=0, weights=detail_weights)

                        if detail_policy is None:
                            search_with_diagnostics = getattr(self.profiles, "search_with_diagnostics", None)
                            if callable(search_with_diagnostics):
                                search = search_with_diagnostics(
                                    combined_embedding,
                                    threshold=self.config.match_threshold,
                                    exclude_person_ids=used_person_ids_this_frame,
                                )
                                match = search.match
                                search_payload = {
                                    "best_match_person_id": search.best_person_id,
                                    "best_match_score": search.best_score,
                                    "match_threshold": float(self.config.match_threshold),
                                    "match_reason": search.reason,
                                    "eligible_profile_count": search.eligible_profile_count,
                                    "excluded_person_ids": list(search.excluded_person_ids),
                                }
                            else:
                                match = self.profiles.search(
                                    combined_embedding,
                                    threshold=self.config.match_threshold,
                                    exclude_person_ids=used_person_ids_this_frame,
                                )
                                search_payload = {
                                    "best_match_person_id": match.person_id if match is not None else None,
                                    "best_match_score": float(match.score) if match is not None else None,
                                    "match_threshold": float(self.config.match_threshold),
                                    "match_reason": (
                                        "matched_existing_profile"
                                        if match is not None else "profile_manager_rejected"
                                    ),
                                    "eligible_profile_count": None,
                                    "excluded_person_ids": sorted(used_person_ids_this_frame),
                                }
                            created_identity = match is None
                            if created_identity:
                                person_id = self.profiles.create_person_id()
                                score = None
                                created_persons += 1
                            else:
                                person_id = match.person_id
                                score = match.score
                                matched_events += 1
                            decision_zone = "new" if created_identity else "matched"
                            should_store_update = True
                            state = "created_identity" if created_identity else "matched_identity"
                        else:
                            profiles = list(self.store.iter_profiles())
                            matches = detail_policy.rank_profiles(
                                combined_embedding, combined_detail_vector, profiles,
                                exclude_person_ids=used_person_ids_this_frame,
                                bbox_xyxy=best_candidate.bbox_xyxy, frame_index=best_candidate.frame_index,
                                image_width=width, image_height=height,
                                recent_person_positions=recent_person_positions,
                            )
                            match = matches[0] if matches else None
                            decision_zone = "empty_db" if not profiles else (
                                match.decision_zone if match is not None else "low"
                            )
                            search_payload = {
                                "best_match_person_id": match.person_id if match is not None else None,
                                "best_match_score": float(match.score) if match is not None else None,
                                "match_threshold": float(self.config.strong_match_threshold),
                                "match_reason": f"details_{decision_zone}",
                                "eligible_profile_count": sum(
                                    profile.person_id not in used_person_ids_this_frame
                                    for profile in profiles
                                ),
                                "excluded_person_ids": sorted(used_person_ids_this_frame),
                            }
                            created_identity = False
                            should_store_update = False
                            state = "pending_new_person"
                            if decision_zone == "empty_db":
                                person_id = self.profiles.create_person_id()
                                score = None
                                created_persons += 1
                                created_identity = True
                                should_store_update = True
                                state = "created_identity"
                            elif match is not None and decision_zone == "strong":
                                person_id, score = match.person_id, match.score
                                matched_events += 1
                                strong_match_events += 1
                                should_store_update = True
                                state = "matched_identity"
                            elif match is not None and decision_zone == "weak":
                                person_id, score = match.person_id, match.score
                                matched_events += 1
                                pending_weak_match_events += 1
                                state = "pending_weak_match"
                            else:
                                best_score = float(match.score) if match is not None else -1.0
                                overlaps = detail_policy.overlap_protection_enabled and any(
                                    intersection_over_smaller_box(best_candidate.bbox_xyxy, assigned_box)
                                    >= detail_policy.new_person_overlap_threshold
                                    for _, assigned_box in assigned_person_boxes_this_frame
                                )
                                if overlaps:
                                    state = "pending_overlapping_detection"
                                elif not detail_policy.delayed_new_person_enabled:
                                    person_id = self.profiles.create_person_id()
                                    score = None
                                    created_persons += 1
                                    created_identity = True
                                    should_store_update = True
                                    state = "created_identity"
                                else:
                                    evidence = track_low_match_evidence.setdefault(detection.track_id, [])
                                    evidence.append((frame_index, best_score))
                                    minimum_frame = frame_index - detail_policy.new_person_evidence_window_frames
                                    evidence[:] = [(idx, value) for idx, value in evidence if idx >= minimum_frame]
                                    if has_sufficient_new_person_evidence(
                                        evidence, max_score=detail_policy.new_person_max_score,
                                        min_events=detail_policy.new_person_min_evidence_events,
                                        min_span_frames=detail_policy.new_person_min_evidence_span_frames,
                                        low_match_ratio=detail_policy.new_person_low_match_ratio,
                                    ):
                                        person_id = self.profiles.create_person_id()
                                        score = None
                                        created_persons += 1
                                        created_identity = True
                                        should_store_update = True
                                        state = "created_identity"
                                        track_low_match_evidence[detection.track_id] = []
                                    else:
                                        person_id = None
                                        score = best_score if match is not None else None
                                        pending_new_person_events += 1

                            if match is not None:
                                prediction.update(
                                    match_visual_score=match.visual_score,
                                    match_detail_score=match.detail_score,
                                    match_detail_weight=match.detail_weight,
                                    match_motion_bonus=match.motion_bonus,
                                )

                        prediction.update(search_payload)
                        if person_id is None:
                            prediction.update(state=state, match_score=score, decision_zone=decision_zone,
                                              decision_frame_index=frame_index)
                            if self.config.draw_debug:
                                draw_detection(display_frame, detection, None, score)
                            continue

                        decision = None
                        snapshot_path = None
                        if should_store_update:
                            snapshot_path = save_crop(
                                best_candidate.crop, self.paths.snapshot_dir, person_id,
                                best_candidate.frame_index, run_id=run_id,
                            )
                            quality_payload = self._quality_payload(
                                run_id=run_id,
                                event_type="initial_buffer_match",
                                candidate=best_candidate,
                                good_frame_count=len(candidates),
                                quality_average=quality_average,
                                decision_frame_index=frame_index,
                                detail_vector=combined_detail_vector,
                                match=match,
                            )
                            quality_payload.update(search_payload)
                            quality_payload["snapshot_path_relative_to_artifact_root"] = (
                                artifacts.relative_path(snapshot_path)
                            )
                            decision = self.profiles.add_or_update_person(
                                person_id=person_id, embedding=combined_embedding,
                                source=self._source_to_label(source), frame_index=frame_index,
                                track_id=detection.track_id, bbox_xyxy=best_candidate.bbox_xyxy,
                                score=score, snapshot_path=str(snapshot_path),
                                payload=quality_payload,
                                batch=self._embedding_batch(candidates),
                            )

                        track_to_person[detection.track_id] = person_id
                        track_to_last_score[detection.track_id] = score
                        track_is_tentative[detection.track_id] = state == "pending_weak_match"
                        track_candidates[detection.track_id] = []
                        track_candidate_last_frame.pop(detection.track_id, None)
                        prediction.update(person_id=person_id, match_score=score, state=state,
                                          decision_zone=decision_zone, decision_frame_index=frame_index,
                                          snapshot_frame_index=best_candidate.frame_index if snapshot_path else None)
                        if decision is not None:
                            prediction.update(profile_update_accepted=decision.accepted,
                                              update_similarity=decision.similarity,
                                              profile_update_reason=decision.reason)
                            if not decision.accepted:
                                prediction["state"] = "matched_identity_update_rejected"

                    # Known tracks have already passed both quality gates;
                    # the profile service still checks embedding similarity.
                    else:
                        if detail_policy is not None and track_is_tentative.get(detection.track_id, False):
                            prediction.update(state="pending_weak_match", decision_zone="weak")
                        else:
                            snapshot_path = save_crop(candidate.crop, self.paths.snapshot_dir, person_id, frame_index, run_id=run_id)
                            quality_payload = self._quality_payload(
                                run_id=run_id,
                                event_type="quality_gated_update",
                                candidate=candidate,
                                good_frame_count=1,
                                decision_frame_index=frame_index,
                            )
                            quality_payload["snapshot_path_relative_to_artifact_root"] = (
                                artifacts.relative_path(snapshot_path)
                            )
                            decision = self.profiles.add_or_update_person(
                                person_id=person_id,
                                embedding=normalize_vector(candidate.embedding),
                                source=self._source_to_label(source),
                                frame_index=frame_index,
                                track_id=detection.track_id,
                                bbox_xyxy=detection.bbox_xyxy,
                                score=score,
                                snapshot_path=str(snapshot_path),
                                payload=quality_payload,
                            )
                            prediction.update(state="profile_update" if decision.accepted else "profile_update_rejected",
                                              snapshot_frame_index=frame_index, profile_update_accepted=decision.accepted,
                                              update_similarity=decision.similarity, profile_update_reason=decision.reason)

                if person_id:
                    used_person_ids_this_frame.add(person_id)
                    recent_person_positions[person_id] = (frame_index, bbox_center(detection.bbox_xyxy))
                    assigned_person_boxes_this_frame.append((person_id, detection.bbox_xyxy))

                if self.config.draw_debug:
                    draw_detection(
                        display_frame,
                        detection,
                        person_id,
                        score,
                    )

            # Export and UI callbacks see the fully annotated frame.
            writer.write(display_frame)
            artifacts.record_frame(frame_index, fps, frame_predictions)

            if frame_callback and (
                frame_index == 1 or frame_index % max(1, self.config.live_preview_every_n_frames) == 0
            ):
                frame_callback(frame_index, display_frame)

            if progress_callback and (frame_index == 1 or frame_index % 10 == 0):
                progress_callback(
                    frame_index,
                    total_for_progress,
                    (
                        f"[{self.config.mode_id}] Processed frame {frame_index} | "
                        f"detections: {len(detections)} | skipped low quality: {skipped_low_quality}"
                    ),
                )

        if frame_index == 0:
            raise RuntimeError("The source contained no decodable frames.")
        expected_frames = min(total_frames, self.config.max_frames) if total_frames and self.config.max_frames > 0 else total_frames
        if isinstance(source, str) and Path(source).is_file() and expected_frames and frame_index < expected_frames:
            raise RuntimeError(f"Video ended early: processed {frame_index} of {expected_frames} declared/requested frames. Check decoding/container metadata.")

        if progress_callback:
            progress_callback(frame_index, total_for_progress, f"[{self.config.mode_id}] Finished")

        if skipped_low_quality > 0:
            warnings.append(f"Skipped low-quality ReID crops: {skipped_low_quality}")
        if skipped_overlap > 0:
            warnings.append(f"Skipped ReID crops due to person overlap or cooldown: {skipped_overlap}")
        if expired_track_states > 0:
            warnings.append(f"Expired run-local track states after absence: {expired_track_states}")
        artifacts.add_runtime_summary(expired_track_states=expired_track_states)
        if waiting_for_good_frames > 0:
            warnings.append(
                "Some tracks were not assigned immediately because the pipeline waited for enough good frames."
            )
        if pending_weak_match_events > 0:
            warnings.append(f"Tentative weak matches without profile update: {pending_weak_match_events}")
        if pending_new_person_events > 0:
            warnings.append(f"Low-score observations kept pending before new-person creation: {pending_new_person_events}")

        persons = self.profiles.list_persons()
        return PipelineResult(
            output_video_path=output_path,
            processed_frames=frame_index,
            created_persons=created_persons,
            matched_events=matched_events,
            persons=persons,
            warnings=warnings,
            run_id=run_id,
            mode_id=self.config.mode_id,
            mode_name=self.config.mode_name,
            pipeline_type=self.config.pipeline_type,
            predictions_path=artifacts.predictions_path,
            tracking_predictions_path=artifacts.tracking_path,
            manifest_path=artifacts.manifest_path,
            decision_policy="details_tracking_v2" if detail_policy is not None else "main_single_threshold",
            strong_match_events=strong_match_events,
            pending_weak_match_events=pending_weak_match_events,
            pending_new_person_events=pending_new_person_events,
        )
