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
from app.pipeline.reid_encoder import build_encoder
from app.reid.repository import EmbeddingBatch, EvaluationRepository, ProfileManager, ProfileMatcher, ProfileUpdater
from app.reid.service import ProfileService
from app.reid.embeddings import checked_embedding
from app.storage.encoder_paths import paths_for_encoder
from app.storage.models import Detection, PipelineResult
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


@dataclass
class PipelineRunState:
    """Mutable state whose lifetime is exactly one processed source."""

    track_to_person: dict[int, str]
    track_to_last_score: dict[int, float | None]
    track_candidates: dict[int, list[TrackEmbeddingCandidate]]
    overlap_cooldown_until: dict[int, int]
    created_persons: int = 0
    matched_events: int = 0
    frame_index: int = 0
    skipped_low_quality: int = 0
    skipped_overlap: int = 0
    waiting_for_good_frames: int = 0

    @classmethod
    def empty(cls) -> "PipelineRunState":
        return cls({}, {}, {}, {})


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
        return payload

    def _overlap_by_detection(
        self,
        detections: list[Detection],
        state: PipelineRunState,
    ) -> dict[int, float]:
        """Measure pairwise overlap and extend cooldowns for affected tracks."""

        overlap_by_detection = {index: 0.0 for index in range(len(detections))}
        for first_index, first in enumerate(detections):
            for second_index in range(first_index + 1, len(detections)):
                second = detections[second_index]
                if first.track_id is not None and first.track_id == second.track_id:
                    continue
                overlap = person_bbox_overlap_ratio(first.bbox_xyxy, second.bbox_xyxy)
                overlap_by_detection[first_index] = max(overlap_by_detection[first_index], overlap)
                overlap_by_detection[second_index] = max(overlap_by_detection[second_index], overlap)
                if overlap <= self.config.max_person_overlap_ratio:
                    continue
                cooldown_until = state.frame_index + self.config.overlap_cooldown_frames
                for track_id in (first.track_id, second.track_id):
                    if track_id is not None:
                        state.overlap_cooldown_until[track_id] = max(
                            state.overlap_cooldown_until.get(track_id, 0), cooldown_until
                        )
        return overlap_by_detection

    def _quality_rejection(
        self,
        quality_score: float,
        quality_details: dict[str, float],
        person_id: str | None,
    ) -> str | None:
        """Return the first failed quality gate in evaluation order."""

        if quality_score < self.config.min_embedding_quality:
            return "below_candidate_quality"
        if person_id is None and quality_details.get("blur", 1.0) < self.config.min_initial_blur_score:
            return "below_initial_blur"
        if (
            person_id is None
            and quality_details.get("edge_cutoff", 1.0) < 1.0
            and quality_details.get("blur", 1.0) < self.config.min_border_blur_score
        ):
            return "below_border_blur"
        if (
            person_id is None
            and quality_details.get("aspect_ratio", 1.0) < self.config.min_initial_aspect_ratio_score
        ):
            return "below_initial_aspect_ratio"
        if person_id is not None and quality_score < self.config.min_update_quality:
            return "below_update_quality"
        return None

    def _encode_candidate(
        self,
        crop: np.ndarray,
        detection: Detection,
        quality_score: float,
        quality_details: dict[str, float],
        frame_index: int,
    ) -> TrackEmbeddingCandidate:
        """Encode one accepted crop and enforce a stable run-wide dimension."""

        assert self.encoder is not None
        embedding = checked_embedding(self.encoder.encode(crop))
        if hasattr(self, "_embedding_dimension") and embedding.size != self._embedding_dimension:
            raise ValueError("The encoder changed its embedding dimension within a run.")
        self._embedding_dimension = embedding.size
        return TrackEmbeddingCandidate(
            embedding=embedding,
            quality_score=quality_score,
            quality_details=quality_details,
            crop=crop.copy(),
            bbox_xyxy=detection.bbox_xyxy,
            frame_index=frame_index,
            detection_confidence=detection.confidence,
        )

    def _search_profiles(
        self,
        embedding: np.ndarray,
        used_person_ids: set[str],
    ) -> tuple[Any, dict[str, object]]:
        """Search profiles while preserving diagnostics for older managers."""

        search_with_diagnostics = getattr(self.profiles, "search_with_diagnostics", None)
        if callable(search_with_diagnostics):
            search = search_with_diagnostics(
                embedding,
                threshold=self.config.match_threshold,
                exclude_person_ids=used_person_ids,
            )
            return search.match, {
                "best_match_person_id": search.best_person_id,
                "best_match_score": search.best_score,
                "match_threshold": float(self.config.match_threshold),
                "match_reason": search.reason,
                "eligible_profile_count": search.eligible_profile_count,
                "excluded_person_ids": list(search.excluded_person_ids),
            }

        match = self.profiles.search(
            embedding,
            threshold=self.config.match_threshold,
            exclude_person_ids=used_person_ids,
        )
        return match, {
            "best_match_person_id": match.person_id if match is not None else None,
            "best_match_score": float(match.score) if match is not None else None,
            "match_threshold": float(self.config.match_threshold),
            "match_reason": "matched_existing_profile" if match is not None else "profile_manager_rejected",
            "eligible_profile_count": None,
            "excluded_person_ids": sorted(used_person_ids),
        }

    def _initial_identity_decision(
        self,
        *,
        detection: Detection,
        candidates: list[TrackEmbeddingCandidate],
        used_person_ids: set[str],
        run_id: str,
        source: str | int | CameraSource,
        artifacts: RunArtifacts,
        state: PipelineRunState,
        prediction: dict[str, object],
    ) -> str:
        """Match or create one identity after its initial buffer is complete."""

        combined_embedding = self._combined_embedding(candidates)
        best_candidate = self._best_candidate(candidates)
        quality_average = float(np.mean([item.quality_score for item in candidates]))
        match, search_payload = self._search_profiles(combined_embedding, used_person_ids)
        if match is None:
            person_id = self.profiles.create_person_id()
            score = None
            state.created_persons += 1
        else:
            person_id = match.person_id
            score = match.score
            state.matched_events += 1

        snapshot_path = save_crop(
            best_candidate.crop,
            self.paths.snapshot_dir,
            person_id,
            best_candidate.frame_index,
            run_id=run_id,
        )
        quality_payload = self._quality_payload(
            run_id=run_id,
            event_type="initial_buffer_match",
            candidate=best_candidate,
            good_frame_count=len(candidates),
            quality_average=quality_average,
            decision_frame_index=state.frame_index,
        )
        quality_payload.update(search_payload)
        quality_payload["snapshot_path_relative_to_artifact_root"] = artifacts.relative_path(snapshot_path)
        decision = self.profiles.add_or_update_person(
            person_id=person_id,
            embedding=combined_embedding,
            source=self._source_to_label(source),
            frame_index=state.frame_index,
            track_id=detection.track_id,
            bbox_xyxy=best_candidate.bbox_xyxy,
            score=score,
            snapshot_path=str(snapshot_path),
            payload=quality_payload,
            batch=self._embedding_batch(candidates),
        )

        assert detection.track_id is not None
        state.track_to_person[detection.track_id] = person_id
        state.track_to_last_score[detection.track_id] = score
        state.track_candidates[detection.track_id] = []
        prediction.update(
            person_id=person_id,
            match_score=score,
            state="created_identity" if match is None else "matched_identity",
            decision_frame_index=state.frame_index,
            snapshot_frame_index=best_candidate.frame_index,
        )
        prediction.update(search_payload)
        prediction.update(
            profile_update_accepted=decision.accepted,
            update_similarity=decision.similarity,
            profile_update_reason=decision.reason,
        )
        if not decision.accepted:
            prediction["state"] = "matched_identity_update_rejected"
        return person_id

    def _update_known_identity(
        self,
        *,
        detection: Detection,
        candidate: TrackEmbeddingCandidate,
        person_id: str,
        score: float | None,
        run_id: str,
        source: str | int | CameraSource,
        artifacts: RunArtifacts,
        state: PipelineRunState,
        prediction: dict[str, object],
    ) -> None:
        """Persist one quality-approved observation for a known track."""

        snapshot_path = save_crop(
            candidate.crop,
            self.paths.snapshot_dir,
            person_id,
            state.frame_index,
            run_id=run_id,
        )
        quality_payload = self._quality_payload(
            run_id=run_id,
            event_type="quality_gated_update",
            candidate=candidate,
            good_frame_count=1,
            decision_frame_index=state.frame_index,
        )
        quality_payload["snapshot_path_relative_to_artifact_root"] = artifacts.relative_path(snapshot_path)
        decision = self.profiles.add_or_update_person(
            person_id=person_id,
            embedding=normalize_vector(candidate.embedding),
            source=self._source_to_label(source),
            frame_index=state.frame_index,
            track_id=detection.track_id,
            bbox_xyxy=detection.bbox_xyxy,
            score=score,
            snapshot_path=str(snapshot_path),
            payload=quality_payload,
        )
        prediction.update(
            state="profile_update" if decision.accepted else "profile_update_rejected",
            snapshot_frame_index=state.frame_index,
            profile_update_accepted=decision.accepted,
            update_similarity=decision.similarity,
            profile_update_reason=decision.reason,
        )

    def _prediction_for_detection(
        self,
        detection: Detection,
        person_id: str | None,
        score: float | None,
        overlap_ratio: float,
        state: PipelineRunState,
    ) -> dict[str, object]:
        """Create the stable frame-export schema before applying decisions."""

        return {
            "bbox_xyxy": list(detection.bbox_xyxy), "confidence": float(detection.confidence),
            "class_id": detection.class_id, "track_id": detection.track_id, "person_id": person_id,
            "match_score": score, "state": "known_track" if person_id else "pending",
            "quality_score": None, "decision_frame_index": None, "snapshot_frame_index": None,
            "profile_update_accepted": None, "update_similarity": None, "profile_update_reason": None,
            "person_overlap_ratio": overlap_ratio,
            "overlap_cooldown_until_frame": state.overlap_cooldown_until.get(detection.track_id),
            "overlap_cooldown_remaining": max(
                0, state.overlap_cooldown_until.get(detection.track_id, state.frame_index) - state.frame_index
            ),
            "quality_details": None,
            "initial_candidate_count": (
                len(state.track_candidates.get(detection.track_id, []))
                if detection.track_id is not None and person_id is None else None
            ),
            "initial_candidate_required": (
                self.config.min_good_frames_before_reid
                if detection.track_id is not None and person_id is None else None
            ),
            "best_match_person_id": None, "best_match_score": None, "match_threshold": None,
            "match_reason": None, "eligible_profile_count": None, "excluded_person_ids": None,
        }

    def _process_detection(
        self,
        *,
        detection: Detection,
        overlap_ratio: float,
        frame: np.ndarray,
        display_frame: np.ndarray,
        used_person_ids: set[str],
        run_id: str,
        source: str | int | CameraSource,
        artifacts: RunArtifacts,
        state: PipelineRunState,
    ) -> dict[str, object]:
        """Apply overlap, crop, quality, matching and update rules to one box."""

        person_id = state.track_to_person.get(detection.track_id)
        score = state.track_to_last_score.get(detection.track_id)
        prediction = self._prediction_for_detection(detection, person_id, score, overlap_ratio, state)

        def draw() -> None:
            if self.config.draw_debug:
                draw_detection(display_frame, detection, person_id, score)

        if detection.track_id is None:
            prediction["state"] = "untracked"
            draw()
            return prediction

        currently_overlapping = overlap_ratio > self.config.max_person_overlap_ratio
        in_overlap_cooldown = state.frame_index <= state.overlap_cooldown_until.get(detection.track_id, 0)
        if person_id is None and currently_overlapping:
            # Discard pre-crossing views because a tracker ID may switch owners.
            state.track_candidates.pop(detection.track_id, None)
            prediction["initial_candidate_count"] = 0

        should_reid = person_id is None or state.frame_index % self.config.reid_every_n_frames == 0
        if should_reid and (currently_overlapping or in_overlap_cooldown):
            prediction["state"] = "overlapping_person" if currently_overlapping else "overlap_cooldown"
            state.skipped_overlap += 1
            draw()
            return prediction

        if should_reid:
            crop = crop_xyxy(frame, detection.bbox_xyxy, padding=self.config.crop_padding)
            if not is_valid_crop(crop, self.config.min_crop_width, self.config.min_crop_height):
                prediction["state"] = "invalid_crop"
                state.skipped_low_quality += 1
                draw()
                return prediction

            quality_score, quality_details = crop_quality_score(
                crop=crop,
                bbox_xyxy=detection.bbox_xyxy,
                frame_shape=frame.shape,
                detection_confidence=detection.confidence,
                min_width=self.config.min_crop_width,
                min_height=self.config.min_crop_height,
            )
            prediction["quality_score"] = quality_score
            prediction["quality_details"] = {key: float(value) for key, value in quality_details.items()}
            rejection = self._quality_rejection(quality_score, quality_details, person_id)
            if rejection is not None:
                prediction["state"] = rejection
                state.skipped_low_quality += 1
                draw()
                return prediction

            candidate = self._encode_candidate(
                crop, detection, quality_score, quality_details, state.frame_index
            )
            if person_id is None:
                candidates = state.track_candidates.setdefault(detection.track_id, [])
                candidates.append(candidate)
                max_buffer_size = max(self.config.min_good_frames_before_reid * 2, 5)
                if len(candidates) > max_buffer_size:
                    del candidates[0:len(candidates) - max_buffer_size]
                prediction["initial_candidate_count"] = len(candidates)
                if len(candidates) < self.config.min_good_frames_before_reid:
                    prediction["state"] = "waiting_for_initial_observations"
                    state.waiting_for_good_frames += 1
                    draw()
                    return prediction
                person_id = self._initial_identity_decision(
                    detection=detection,
                    candidates=candidates,
                    used_person_ids=used_person_ids,
                    run_id=run_id,
                    source=source,
                    artifacts=artifacts,
                    state=state,
                    prediction=prediction,
                )
                score = state.track_to_last_score[detection.track_id]
            else:
                self._update_known_identity(
                    detection=detection,
                    candidate=candidate,
                    person_id=person_id,
                    score=score,
                    run_id=run_id,
                    source=source,
                    artifacts=artifacts,
                    state=state,
                    prediction=prediction,
                )

        if person_id:
            used_person_ids.add(person_id)
        draw()
        return prediction

    def _process_frame(
        self,
        *,
        frame: np.ndarray,
        run_id: str,
        source: str | int | CameraSource,
        artifacts: RunArtifacts,
        state: PipelineRunState,
    ) -> tuple[np.ndarray, list[Detection], list[dict[str, object]]]:
        """Track and process every detection in one unmodified source frame."""

        assert self.tracker is not None
        detections = self.tracker.track_frame(frame)
        overlap_by_detection = self._overlap_by_detection(detections, state)
        display_frame = frame.copy() if self.config.draw_debug else frame
        # Reserve visible known IDs before unknown tracks search the same profiles.
        used_person_ids = {
            state.track_to_person[detection.track_id]
            for detection in detections
            if detection.track_id in state.track_to_person
        }
        predictions = [
            self._process_detection(
                detection=detection,
                overlap_ratio=overlap_by_detection[index],
                frame=frame,
                display_frame=display_frame,
                used_person_ids=used_person_ids,
                run_id=run_id,
                source=source,
                artifacts=artifacts,
                state=state,
            )
            for index, detection in enumerate(detections)
        ]
        return display_frame, detections, predictions

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

        if getattr(self, "_has_processed", False):
            raise RuntimeError(
                "A pipeline instance handles one source only; construct a fresh tracker/pipeline for the next run."
            )
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
                        device=self.config.device, model_name=self.config.reid_model_name,
                        checkpoint_path=self.config.reid_checkpoint,
                    )
                if isinstance(self.encoder, ReleasableBackend) and self.encoder is not self.tracker:
                    resources.append(self.encoder)
            finally:
                load_seconds = time.perf_counter() - load_started
            components: dict[str, dict[str, Any]] = {}
            for label, backend in (("tracker", self.tracker), ("encoder", self.encoder), ("profiles", self.profiles)):
                implementation = f"{type(backend).__module__}.{type(backend).__qualname__}"
                description: dict[str, Any] = {"implementation": implementation}
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
        artifacts.metadata["exports"]["annotated_video_relative_to_artifact_root"] = artifacts.relative_path(
            output_path
        )
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

        state = PipelineRunState.empty()
        warnings: list[str] = []

        # Hot path: read -> track -> quality gate -> encode/match -> draw/write.
        capture_started = time.perf_counter()
        while True:
            if max_duration_seconds is not None:
                if state.frame_index > 0 and time.perf_counter() - capture_started >= max_duration_seconds:
                    break
            elif self.config.max_frames > 0 and state.frame_index >= self.config.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            state.frame_index += 1
            display_frame, detections, frame_predictions = self._process_frame(
                frame=frame,
                run_id=run_id,
                source=source,
                artifacts=artifacts,
                state=state,
            )

            # Export and UI callbacks see the fully annotated frame.
            writer.write(display_frame)
            artifacts.record_frame(state.frame_index, fps, frame_predictions)

            if frame_callback and (
                state.frame_index == 1
                or state.frame_index % max(1, self.config.live_preview_every_n_frames) == 0
            ):
                frame_callback(state.frame_index, display_frame)

            if progress_callback and (state.frame_index == 1 or state.frame_index % 10 == 0):
                progress_callback(
                    state.frame_index,
                    total_for_progress,
                    (
                        f"[{self.config.mode_id}] Processed frame {state.frame_index} | "
                        f"detections: {len(detections)} | skipped low quality: {state.skipped_low_quality}"
                    ),
                )

        if state.frame_index == 0:
            raise RuntimeError("The source contained no decodable frames.")
        expected_frames = (
            min(total_frames, self.config.max_frames)
            if total_frames and self.config.max_frames > 0
            else total_frames
        )
        ended_early = (
            isinstance(source, str)
            and Path(source).is_file()
            and expected_frames
            and state.frame_index < expected_frames
        )
        if ended_early:
            raise RuntimeError(
                f"Video ended early: processed {state.frame_index} of {expected_frames} declared/requested frames. "
                "Check decoding/container metadata."
            )

        if progress_callback:
            progress_callback(state.frame_index, total_for_progress, f"[{self.config.mode_id}] Finished")

        if state.skipped_low_quality > 0:
            warnings.append(f"Skipped low-quality ReID crops: {state.skipped_low_quality}")
        if state.skipped_overlap > 0:
            warnings.append(f"Skipped ReID crops due to person overlap or cooldown: {state.skipped_overlap}")
        if state.waiting_for_good_frames > 0:
            warnings.append(
                "Some tracks were not assigned immediately because the pipeline waited for enough good frames."
            )

        persons = self.profiles.list_persons()
        return PipelineResult(
            output_video_path=output_path,
            processed_frames=state.frame_index,
            created_persons=state.created_persons,
            matched_events=state.matched_events,
            persons=persons,
            warnings=warnings,
            run_id=run_id,
            mode_id=self.config.mode_id,
            mode_name=self.config.mode_name,
            pipeline_type=self.config.pipeline_type,
            predictions_path=artifacts.predictions_path,
            tracking_predictions_path=artifacts.tracking_path,
            manifest_path=artifacts.manifest_path,
        )
