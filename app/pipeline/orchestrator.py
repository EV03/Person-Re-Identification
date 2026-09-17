from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np

from app.config import AppPaths, PipelineConfig
from app.pipeline.detector_tracker import UltralyticsPersonTracker
from app.pipeline.reid_encoder import build_encoder
from app.storage.models import MatchResult, PipelineResult
from app.utils.detail_utils import DetailSnapshot, compact_detail_influences, extract_detail_snapshot
from app.storage.store_factory import build_vector_store
from app.utils.camera_utils import CameraSource, open_camera_capture
from app.utils.id_utils import make_run_id, safe_source_name
from app.utils.motion_utils import MotionSnapshot, MotionTracker
from app.utils.image_utils import (
    crop_quality_score,
    crop_xyxy,
    draw_detection,
    is_valid_crop,
    normalize_vector,
    save_crop,
)

ProgressCallback = Callable[[int, int | None, str], None]
FrameCallback = Callable[[int, np.ndarray], None]
EventCallback = Callable[[dict[str, object]], None]


def intersection_over_smaller_box(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Return how much of the smaller box is covered by the intersection."""

    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    intersection_width = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection_area = intersection_width * intersection_height
    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    smaller_area = min(first_area, second_area)
    if smaller_area <= 0:
        return 0.0
    return float(intersection_area / smaller_area)


def has_sufficient_new_person_evidence(
    evidence: list[tuple[int, float]],
    *,
    max_score: float,
    min_events: int,
    min_span_frames: int,
    low_match_ratio: float,
) -> bool:
    """Require both repeated low scores and enough elapsed frames for a new identity."""

    if not evidence:
        return False
    low_count = sum(1 for _, score in evidence if score <= max_score)
    low_ratio = float(low_count / len(evidence))
    frame_span = int(evidence[-1][0]) - int(evidence[0][0]) + 1
    return len(evidence) >= min_events and frame_span >= min_span_frames and low_ratio >= low_match_ratio


class FfmpegOutputVideoWriter:
    """Small VideoWriter-compatible adapter backed by the bundled FFmpeg binary."""

    def __init__(self, output_path: Path, fps: float, frame_size: tuple[int, int]) -> None:
        self._generator = imageio_ffmpeg.write_frames(
            str(output_path),
            frame_size,
            pix_fmt_in="bgr24",
            pix_fmt_out="yuv420p",
            fps=fps,
            quality=7,
            codec="libx264",
            macro_block_size=2,
            ffmpeg_log_level="error",
            output_params=["-movflags", "+faststart"],
        )
        self._generator.send(None)
        self._is_open = True

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoWriter
        return self._is_open

    def write(self, frame: np.ndarray) -> None:
        if not self._is_open:
            raise RuntimeError("Cannot write to a closed output video.")
        self._generator.send(np.ascontiguousarray(frame))

    def release(self) -> None:
        if self._is_open:
            self._generator.close()
            self._is_open = False


def open_output_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[FfmpegOutputVideoWriter, str]:
    """Open an MP4 writer, preferring browser-compatible H.264 output."""

    try:
        return FfmpegOutputVideoWriter(output_path, fps, frame_size), "h264"
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not create browser-compatible H.264 output video: {output_path}") from exc


@dataclass
class TrackEmbeddingCandidate:
    embedding: np.ndarray
    quality_score: float
    quality_details: dict[str, float]
    crop: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    frame_index: int
    detection_confidence: float
    detail_snapshot: DetailSnapshot | None = None


class PersonReIdPipeline:
    def __init__(self, config: PipelineConfig, paths: AppPaths | None = None) -> None:
        self.config = config
        self.paths = paths or AppPaths()
        self.paths.ensure()
        self.store = build_vector_store(config, self.paths)
        self.tracker = UltralyticsPersonTracker(
            model_name=config.yolo_model,
            tracker=config.tracker,
            confidence=config.detection_confidence,
            image_size=config.image_size,
            device=config.device,
        )
        self.encoder = build_encoder(
            model_name=config.reid_model_name,
            model_path=config.reid_model_path,
            device=config.device,
        )

    def close(self) -> None:
        store_close = getattr(self.store, "close", None)
        if callable(store_close):
            store_close()

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
        if not candidates:
            raise ValueError("Cannot build an embedding from an empty candidate list.")
        embeddings = np.stack([normalize_vector(candidate.embedding) for candidate in candidates], axis=0)
        weights = np.asarray([max(candidate.quality_score, 0.05) for candidate in candidates], dtype=np.float32)
        combined = np.average(embeddings, axis=0, weights=weights)
        return normalize_vector(combined)

    @staticmethod
    def _best_candidate(candidates: list[TrackEmbeddingCandidate]) -> TrackEmbeddingCandidate:
        if not candidates:
            raise ValueError("Cannot select a best candidate from an empty candidate list.")
        return max(candidates, key=lambda candidate: candidate.quality_score)

    def _mode_warning(self) -> str | None:
        if self.config.pipeline_type != "football_analysis":
            return None
        return (
            "Football mode v1 uses the stable Default ReID pipeline. "
            "Ball tracking, pitch mapping, team classification and stats aggregation "
            "are prepared as mode flags/placeholders and must be wired in next."
        )

    def _internal_motion_analysis_enabled(self) -> bool:
        if not self.config.enable_motion_analysis:
            return False
        if self.config.disable_internal_motion_when_botsort and "botsort" in self.config.tracker.lower():
            return False
        return True

    def _quality_payload(
        self,
        *,
        run_id: str,
        event_type: str,
        candidate: TrackEmbeddingCandidate,
        good_frame_count: int,
        quality_average: float | None = None,
        motion: MotionSnapshot | None = None,
        match: MatchResult | None = None,
    ) -> dict[str, object]:
        quality_average = candidate.quality_score if quality_average is None else quality_average
        payload: dict[str, object] = {
            "run_id": run_id,
            "mode_id": self.config.mode_id,
            "pipeline_type": self.config.pipeline_type,
            "event_type": event_type,
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
            payload["detail_vector"] = candidate.detail_snapshot.vector
            payload["detail_reliability"] = float(candidate.detail_snapshot.reliability)
        if motion is not None:
            payload["motion"] = motion.to_payload()
            payload["motion_direction"] = motion.direction_label
            payload["motion_speed_px_per_sec"] = float(motion.speed_px_per_sec)
            payload["motion_plausibility_score"] = float(motion.plausibility_score)
            payload["motion_is_large_jump"] = bool(motion.is_large_jump)
        if match is not None:
            summary = compact_detail_influences(match.detail_breakdown, limit=5)
            payload["match_explanation"] = {
                "person_id": match.person_id,
                "final_score": float(match.score),
                "visual_score": float(match.visual_score) if match.visual_score is not None else None,
                "detail_score": float(match.detail_score) if match.detail_score is not None else None,
                "detail_weight": float(match.detail_weight),
                "detail_breakdown": match.detail_breakdown,
                "summary": summary,
                "decision_zone": match.decision_zone,
                "reference_type": match.reference_type,
                "reference_quality": match.reference_quality,
                "top_matches": match.top_matches,
            }
        return payload

    @staticmethod
    def _emit_event(callback: EventCallback | None, event: dict[str, object]) -> None:
        if callback is None:
            return
        callback(event)

    @staticmethod
    def _bbox_center(bbox_xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox_xyxy
        return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)

    @staticmethod
    def _top_matches_payload(matches: list[MatchResult]) -> list[dict[str, object]]:
        return [
            {
                "rank": idx + 1,
                "person_id": match.person_id,
                "score": float(match.score),
                "visual_score": float(match.visual_score) if match.visual_score is not None else None,
                "detail_score": float(match.detail_score) if match.detail_score is not None else None,
                "decision_zone": match.decision_zone,
                "reference_type": match.reference_type,
                "reference_quality": match.reference_quality,
            }
            for idx, match in enumerate(matches)
        ]

    def _search_candidates(
        self,
        embedding: np.ndarray,
        *,
        exclude_person_ids: set[str],
        detail_vector: np.ndarray | list[float] | None,
        detail_weight: float,
        limit: int = 5,
    ) -> list[MatchResult]:
        search_candidates = getattr(self.store, "search_candidates", None)
        if callable(search_candidates):
            matches = search_candidates(
                embedding,
                exclude_person_ids=exclude_person_ids,
                detail_vector=detail_vector,
                detail_weight=detail_weight,
                limit=limit,
            )
        else:
            match = self.store.search(
                embedding,
                threshold=0.0,
                exclude_person_ids=exclude_person_ids,
                detail_vector=detail_vector,
                detail_weight=detail_weight,
            )
            matches = [] if match is None else [match]
        annotated: list[MatchResult] = []
        for match in matches:
            zone = "low"
            if match.score >= self.config.strong_match_threshold:
                zone = "strong"
            elif match.score >= self.config.weak_match_threshold:
                zone = "weak"
            annotated.append(replace(match, decision_zone=zone))
        top_payload = self._top_matches_payload(annotated)
        return [replace(match, top_matches=top_payload) for match in annotated]

    def _apply_motion_identity_bonus(
        self,
        matches: list[MatchResult],
        *,
        bbox_xyxy: tuple[int, int, int, int],
        frame_index: int,
        image_width: int,
        image_height: int,
        recent_person_positions: dict[str, tuple[int, tuple[float, float]]],
    ) -> list[MatchResult]:
        if not matches or self.config.motion_identity_bonus <= 0:
            return matches
        center = self._bbox_center(bbox_xyxy)
        diag = max(float(np.hypot(image_width, image_height)), 1.0)
        max_distance = self.config.motion_identity_max_distance_fraction
        boosted: list[MatchResult] = []
        for match in matches:
            recent = recent_person_positions.get(match.person_id)
            if recent is None:
                boosted.append(match)
                continue
            last_frame, last_center = recent
            frame_gap = int(frame_index) - int(last_frame)
            if frame_gap < 0 or frame_gap > self.config.motion_identity_max_frame_gap:
                boosted.append(match)
                continue
            distance_fraction = float(np.hypot(center[0] - last_center[0], center[1] - last_center[1]) / diag)
            if distance_fraction <= max_distance:
                new_score = float(np.clip(match.score + self.config.motion_identity_bonus, 0.0, 1.0))
                zone = "low"
                if new_score >= self.config.strong_match_threshold:
                    zone = "strong"
                elif new_score >= self.config.weak_match_threshold:
                    zone = "weak"
                boosted.append(replace(match, score=new_score, decision_zone=zone))
            else:
                boosted.append(match)
        boosted.sort(key=lambda item: item.score, reverse=True)
        top_payload = self._top_matches_payload(boosted)
        return [replace(match, top_matches=top_payload) for match in boosted]

    def process(
        self,
        source: str | int | CameraSource,
        progress_callback: ProgressCallback | None = None,
        frame_callback: FrameCallback | None = None,
        event_callback: EventCallback | None = None,
    ) -> PipelineResult:
        cap = self._open_capture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = total_frames_raw if total_frames_raw > 0 else None
        if self.config.max_frames > 0:
            total_for_progress = min(total_frames or self.config.max_frames, self.config.max_frames)
        else:
            total_for_progress = total_frames

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        run_id = make_run_id()
        source_label = safe_source_name(self._source_to_label(source))
        mode_label = safe_source_name(self.config.mode_id)
        output_path = self.paths.output_dir / f"{mode_label}_{run_id}_{source_label}.mp4"
        writer, output_codec = open_output_video_writer(output_path, fps, (width, height))

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
            metadata={
                "yolo_model": self.config.yolo_model,
                "tracker": self.config.tracker,
                "reid_model_name": self.config.reid_model_name,
                "reid_model_path": self.config.reid_model_path,
                "embedding_dimension": self.encoder.embedding_dim,
                "output_video_path": str(output_path),
                "output_video_codec": output_codec,
                "vector_store_backend": self.config.vector_store_backend,
                "qdrant_url": self.config.qdrant_url if self.config.vector_store_backend == "qdrant" else None,
                "qdrant_collection": self.config.qdrant_collection if self.config.vector_store_backend == "qdrant" else None,
                "qdrant_mode": self.config.qdrant_mode if self.config.vector_store_backend == "qdrant" else None,
                "qdrant_local_path": self.config.qdrant_local_path if self.config.vector_store_backend == "qdrant" else None,
                "qdrant_prefer_grpc": self.config.qdrant_prefer_grpc if self.config.vector_store_backend == "qdrant" else None,
                "match_threshold": self.config.match_threshold,
                "strong_match_threshold": self.config.strong_match_threshold,
                "weak_match_threshold": self.config.weak_match_threshold,
                "new_person_max_score": self.config.new_person_max_score,
                "new_person_min_evidence_events": self.config.new_person_min_evidence_events,
                "new_person_evidence_window_frames": self.config.new_person_evidence_window_frames,
                "new_person_low_match_ratio": self.config.new_person_low_match_ratio,
                "pending_max_age_frames": self.config.pending_max_age_frames,
                "motion_identity_bonus": self.config.motion_identity_bonus,
                "motion_identity_max_frame_gap": self.config.motion_identity_max_frame_gap,
                "motion_identity_max_distance_fraction": self.config.motion_identity_max_distance_fraction,
                "merge_candidate_threshold": self.config.merge_candidate_threshold,
                "merge_candidate_min_events": self.config.merge_candidate_min_events,
                "detection_confidence": self.config.detection_confidence,
                "image_size": self.config.image_size,
                "reid_every_n_frames": self.config.reid_every_n_frames,
                "min_good_frames_before_reid": self.config.min_good_frames_before_reid,
                "min_embedding_quality": self.config.min_embedding_quality,
                "min_update_quality": self.config.min_update_quality,
                "enable_motion_analysis": self.config.enable_motion_analysis,
                "internal_motion_analysis_enabled": self._internal_motion_analysis_enabled(),
                "draw_motion_vectors": self.config.draw_motion_vectors,
                "disable_internal_motion_when_botsort": self.config.disable_internal_motion_when_botsort,
                "motion_max_jump_fraction": self.config.motion_max_jump_fraction,
                "motion_smoothing_alpha": self.config.motion_smoothing_alpha,
                "motion_min_displacement_px": self.config.motion_min_displacement_px,
                "enable_detail_analysis": self.config.enable_detail_analysis,
                "detail_weight": self.config.detail_weight,
                "detail_min_confidence": self.config.detail_min_confidence,
                "draw_detail_labels": self.config.draw_detail_labels,
                "enable_ball_tracking": self.config.enable_ball_tracking,
                "enable_pitch_mapping": self.config.enable_pitch_mapping,
                "enable_team_classification": self.config.enable_team_classification,
                "enable_stats_aggregation": self.config.enable_stats_aggregation,
            },
        )

        track_to_person: dict[int, str] = {}
        track_to_last_score: dict[int, float | None] = {}
        track_is_tentative: dict[int, bool] = {}
        track_candidates: dict[int, list[TrackEmbeddingCandidate]] = {}
        track_low_match_evidence: dict[int, list[tuple[int, float]]] = {}
        recent_person_positions: dict[str, tuple[int, tuple[float, float]]] = {}
        internal_motion_analysis_enabled = self._internal_motion_analysis_enabled()
        motion_tracker = (
            MotionTracker(
                fps=float(fps),
                frame_width=width,
                frame_height=height,
                max_jump_fraction=self.config.motion_max_jump_fraction,
                smoothing_alpha=self.config.motion_smoothing_alpha,
                min_displacement_px=self.config.motion_min_displacement_px,
            )
            if internal_motion_analysis_enabled
            else None
        )
        created_persons = 0
        matched_events = 0
        strong_match_events = 0
        pending_weak_match_events = 0
        frame_index = 0
        skipped_low_quality = 0
        waiting_for_good_frames = 0
        overlap_suppressed_events = 0
        large_motion_jumps = 0
        detail_observations = 0
        calibration_mode = str(self.config.calibration_mode or "off")
        calibration_target_person_id = str(self.config.calibration_target_person_id or "").strip()
        calibration_active_person_id = calibration_target_person_id if calibration_mode == "extend_person" and calibration_target_person_id else None
        warnings: list[str] = []
        mode_warning = self._mode_warning()
        if mode_warning:
            warnings.append(mode_warning)
        if (
            self.config.enable_motion_analysis
            and not internal_motion_analysis_enabled
            and self.config.disable_internal_motion_when_botsort
            and "botsort" in self.config.tracker.lower()
        ):
            warnings.append("Internal direction analysis disabled because BoT-SORT is active.")

        while True:
            if self.config.max_frames > 0 and frame_index >= self.config.max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            source_frame = frame.copy()

            detections = self.tracker.track_frame(source_frame)
            used_person_ids_this_frame: set[str] = set()
            assigned_person_boxes_this_frame: list[tuple[str, tuple[int, int, int, int]]] = []

            for detection in detections:
                person_id = track_to_person.get(detection.track_id)
                score = track_to_last_score.get(detection.track_id)
                detail_label = None
                motion = (
                    motion_tracker.update(
                        track_id=detection.track_id,
                        bbox_xyxy=detection.bbox_xyxy,
                        frame_index=frame_index,
                    )
                    if internal_motion_analysis_enabled and motion_tracker is not None
                    else None
                )
                if motion is not None and motion.is_large_jump:
                    large_motion_jumps += 1

                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = detection.bbox_xyxy

                def emit_pipeline_event(event_type: str, **extra: object) -> None:
                    event: dict[str, object] = {
                        "run_id": run_id,
                        "mode_id": self.config.mode_id,
                        "pipeline_type": self.config.pipeline_type,
                        "source": self._source_to_label(source),
                        "source_label": source_label,
                        "frame_index": int(frame_index),
                        "track_id": int(detection.track_id),
                        "person_id": person_id,
                        "match_score": score,
                        "detection_confidence": float(detection.confidence),
                        "bbox_x1": int(bbox_x1),
                        "bbox_y1": int(bbox_y1),
                        "bbox_x2": int(bbox_x2),
                        "bbox_y2": int(bbox_y2),
                        "event_type": event_type,
                    }
                    if motion is not None:
                        event.update(
                            {
                                "motion_direction": motion.direction_label,
                                "motion_speed_px_per_sec": float(motion.speed_px_per_sec),
                                "motion_plausibility_score": float(motion.plausibility_score),
                                "motion_is_large_jump": bool(motion.is_large_jump),
                            }
                        )
                    event.update(extra)
                    self._emit_event(event_callback, event)

                should_reid = person_id is None or frame_index % max(1, self.config.reid_every_n_frames) == 0

                if should_reid:
                    crop = crop_xyxy(source_frame, detection.bbox_xyxy, padding=self.config.crop_padding)
                    if not is_valid_crop(crop, self.config.min_crop_width, self.config.min_crop_height):
                        skipped_low_quality += 1
                        emit_pipeline_event(
                            "invalid_crop",
                            should_reid=True,
                            skip_reason="invalid_crop",
                            created_new_person=False,
                        )
                        if self.config.draw_debug:
                            draw_detection(
                                frame,
                                detection,
                                person_id,
                                score,
                                motion=motion,
                                draw_motion=self.config.draw_motion_vectors and internal_motion_analysis_enabled,
                            )
                        continue

                    quality_score, quality_details = crop_quality_score(
                        crop=crop,
                        bbox_xyxy=detection.bbox_xyxy,
                            frame_shape=source_frame.shape,
                        detection_confidence=detection.confidence,
                        min_width=self.config.min_crop_width,
                        min_height=self.config.min_crop_height,
                    )
                    if quality_score < self.config.min_embedding_quality:
                        skipped_low_quality += 1
                        emit_pipeline_event(
                            "low_quality_crop",
                            should_reid=True,
                            skip_reason="quality_below_min_embedding",
                            quality_score=float(quality_score),
                            quality_details={key: float(value) for key, value in quality_details.items()},
                            created_new_person=False,
                        )
                        if self.config.draw_debug:
                            draw_detection(
                                frame,
                                detection,
                                person_id,
                                score,
                                motion=motion,
                                draw_motion=self.config.draw_motion_vectors and internal_motion_analysis_enabled,
                            )
                        continue

                    embedding = self.encoder.encode(crop)
                    detail_snapshot = (
                        extract_detail_snapshot(crop, min_confidence=self.config.detail_min_confidence)
                        if self.config.enable_detail_analysis
                        else None
                    )
                    if detail_snapshot is not None:
                        detail_observations += 1
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

                    if detail_snapshot is not None and self.config.draw_detail_labels:
                        detail_label = detail_snapshot.label

                    if track_is_tentative.get(detection.track_id, False):
                        person_id = None
                        score = None

                    if person_id is None:
                        candidates = track_candidates.setdefault(detection.track_id, [])
                        candidates.append(candidate)
                        max_buffer_size = max(self.config.min_good_frames_before_reid * 2, 5)
                        if len(candidates) > max_buffer_size:
                            del candidates[0 : len(candidates) - max_buffer_size]

                        if len(candidates) < self.config.min_good_frames_before_reid:
                            waiting_for_good_frames += 1
                            emit_pipeline_event(
                                "waiting_for_good_frames",
                                should_reid=True,
                                quality_score=float(quality_score),
                                quality_details={key: float(value) for key, value in quality_details.items()},
                                good_frame_count=int(len(candidates)),
                                detail_label=detail_snapshot.label if detail_snapshot is not None else None,
                                detail_reliability=float(detail_snapshot.reliability) if detail_snapshot is not None else None,
                                created_new_person=False,
                            )
                            if self.config.draw_debug:
                                draw_detection(
                                    frame,
                                    detection,
                                    person_id,
                                    score,
                                    motion=motion,
                                    draw_motion=self.config.draw_motion_vectors and internal_motion_analysis_enabled,
                                )
                            continue

                        combined_embedding = self._combined_embedding(candidates)
                        best_candidate = self._best_candidate(candidates)
                        quality_average = float(np.mean([item.quality_score for item in candidates]))
                        detail_vectors = [
                            np.asarray(item.detail_snapshot.vector, dtype=np.float32)
                            for item in candidates
                            if item.detail_snapshot is not None
                        ]
                        combined_detail_vector = (
                            np.average(detail_vectors, axis=0).astype(np.float32)
                            if detail_vectors
                            else None
                        )

                        # Normal project calibration mode: the selected/calibrated person is built or extended
                        # directly from high-quality crops. This is intentionally separate from evaluation scripts.
                        # It treats calibration as a high-quality profile-building phase, not as a hard final truth.
                        if calibration_mode in {"new_person", "extend_person"}:
                            if calibration_mode == "new_person" and calibration_active_person_id is None:
                                calibration_active_person_id = self.store.create_person_id()
                                created_persons += 1
                            if calibration_mode == "extend_person" and not calibration_active_person_id:
                                raise RuntimeError("Calibration mode 'extend_person' requires a target person_id.")

                            person_id = calibration_active_person_id
                            score = 1.0
                            matched_events += 1
                            event_type = "calibration_new_profile" if calibration_mode == "new_person" else "calibration_extend_profile"
                            snapshot_path = save_crop(
                                best_candidate.crop,
                                self.paths.snapshot_dir,
                                person_id,
                                best_candidate.frame_index,
                            )
                            self.store.add_or_update_person(
                                person_id=person_id,
                                embedding=combined_embedding,
                                source=self._source_to_label(source),
                                frame_index=best_candidate.frame_index,
                                track_id=detection.track_id,
                                bbox_xyxy=best_candidate.bbox_xyxy,
                                score=score,
                                snapshot_path=str(snapshot_path),
                                payload=self._quality_payload(
                                    run_id=run_id,
                                    event_type=event_type,
                                    candidate=best_candidate,
                                    good_frame_count=len(candidates),
                                    quality_average=quality_average,
                                    motion=motion,
                                    match=None,
                                ) | {
                                    "calibration_mode": calibration_mode,
                                    "calibration_label": self.config.calibration_label or None,
                                    "calibration_target_person_id": calibration_target_person_id or None,
                                },
                            )
                            emit_pipeline_event(
                                event_type,
                                should_reid=True,
                                person_id=person_id,
                                match_score=score,
                                quality_score=float(best_candidate.quality_score),
                                quality_average=float(quality_average),
                                quality_details={key: float(value) for key, value in best_candidate.quality_details.items()},
                                good_frame_count=int(len(candidates)),
                                created_new_person=calibration_mode == "new_person" and created_persons == 1,
                                decision_zone="calibration",
                                calibration_mode=calibration_mode,
                                calibration_label=self.config.calibration_label or None,
                                detail_label=best_candidate.detail_snapshot.label if best_candidate.detail_snapshot is not None else None,
                                detail_reliability=float(best_candidate.detail_snapshot.reliability) if best_candidate.detail_snapshot is not None else None,
                            )
                            track_to_person[detection.track_id] = person_id
                            track_to_last_score[detection.track_id] = score
                            track_is_tentative[detection.track_id] = False
                            recent_person_positions[person_id] = (int(best_candidate.frame_index), self._bbox_center(best_candidate.bbox_xyxy))
                            track_candidates[detection.track_id] = []
                            continue

                        matches = self._search_candidates(
                            combined_embedding,
                            exclude_person_ids=used_person_ids_this_frame,
                            detail_vector=combined_detail_vector,
                            detail_weight=self.config.detail_weight if self.config.enable_detail_analysis else 0.0,
                        )
                        matches = self._apply_motion_identity_bonus(
                            matches,
                            bbox_xyxy=best_candidate.bbox_xyxy,
                            frame_index=best_candidate.frame_index,
                            image_width=width,
                            image_height=height,
                            recent_person_positions=recent_person_positions,
                        )
                        match = matches[0] if matches else None
                        best_score = float(match.score) if match is not None else 0.0
                        decision_zone = "empty_db" if match is None and getattr(self.store, "count_persons", lambda: 0)() == 0 else (match.decision_zone if match is not None else "low")

                        should_create_new = False
                        should_store_update = False
                        event_type = "pending_match"

                        if decision_zone == "empty_db":
                            should_create_new = True
                        elif match is not None and match.score >= self.config.strong_match_threshold:
                            person_id = match.person_id
                            score = match.score
                            matched_events += 1
                            strong_match_events += 1
                            should_store_update = True
                            event_type = "matched_person"
                        elif match is not None and match.score >= self.config.weak_match_threshold:
                            person_id = match.person_id
                            score = match.score
                            matched_events += 1
                            pending_weak_match_events += 1
                            should_store_update = False
                            event_type = "pending_weak_match"
                        else:
                            overlaps_assigned_person = any(
                                intersection_over_smaller_box(best_candidate.bbox_xyxy, assigned_box)
                                >= self.config.new_person_overlap_threshold
                                for _, assigned_box in assigned_person_boxes_this_frame
                            )
                            if overlaps_assigned_person:
                                overlap_suppressed_events += 1
                                event_type = "pending_overlapping_detection"
                            else:
                                evidence = track_low_match_evidence.setdefault(detection.track_id, [])
                                evidence.append((int(frame_index), best_score))
                                min_frame = int(frame_index) - int(self.config.new_person_evidence_window_frames)
                                evidence[:] = [(idx, value) for idx, value in evidence if idx >= min_frame]
                                should_create_new = has_sufficient_new_person_evidence(
                                    evidence,
                                    max_score=self.config.new_person_max_score,
                                    min_events=self.config.new_person_min_evidence_events,
                                    min_span_frames=self.config.new_person_min_evidence_span_frames,
                                    low_match_ratio=self.config.new_person_low_match_ratio,
                                )
                                event_type = "new_person" if should_create_new else "pending_new_person"

                        if should_create_new:
                            person_id = self.store.create_person_id()
                            score = None
                            created_persons += 1
                            should_store_update = True
                            track_low_match_evidence[detection.track_id] = []
                            match = None

                        if not person_id:
                            emit_pipeline_event(
                                event_type,
                                should_reid=True,
                                person_id=None,
                                match_score=best_score if match is not None else None,
                                quality_score=float(best_candidate.quality_score),
                                quality_average=float(quality_average),
                                quality_details={key: float(value) for key, value in best_candidate.quality_details.items()},
                                good_frame_count=int(len(candidates)),
                                created_new_person=False,
                                decision_zone=decision_zone,
                                top_matches=self._top_matches_payload(matches),
                                detail_label=best_candidate.detail_snapshot.label if best_candidate.detail_snapshot is not None else None,
                                detail_reliability=float(best_candidate.detail_snapshot.reliability) if best_candidate.detail_snapshot is not None else None,
                            )
                            continue

                        if should_store_update:
                            snapshot_path = save_crop(
                                best_candidate.crop,
                                self.paths.snapshot_dir,
                                person_id,
                                best_candidate.frame_index,
                            )
                            self.store.add_or_update_person(
                                person_id=person_id,
                                embedding=combined_embedding,
                                source=self._source_to_label(source),
                                frame_index=best_candidate.frame_index,
                                track_id=detection.track_id,
                                bbox_xyxy=best_candidate.bbox_xyxy,
                                score=score,
                                snapshot_path=str(snapshot_path),
                                payload=self._quality_payload(
                                    run_id=run_id,
                                    event_type="initial_buffer_match" if event_type in {"matched_person", "new_person"} else event_type,
                                    candidate=best_candidate,
                                    good_frame_count=len(candidates),
                                    quality_average=quality_average,
                                    motion=motion,
                                    match=match,
                                ),
                            )

                        emit_pipeline_event(
                            event_type,
                            should_reid=True,
                            person_id=person_id,
                            match_score=score,
                            quality_score=float(best_candidate.quality_score),
                            quality_average=float(quality_average),
                            quality_details={key: float(value) for key, value in best_candidate.quality_details.items()},
                            good_frame_count=int(len(candidates)),
                            created_new_person=event_type == "new_person",
                            decision_zone=decision_zone,
                            top_matches=self._top_matches_payload(matches),
                            match_visual_score=float(match.visual_score) if match is not None and match.visual_score is not None else None,
                            match_detail_score=float(match.detail_score) if match is not None and match.detail_score is not None else None,
                            match_detail_weight=float(match.detail_weight) if match is not None else 0.0,
                            match_reason=compact_detail_influences(match.detail_breakdown, limit=5) if match is not None else None,
                            detail_label=best_candidate.detail_snapshot.label if best_candidate.detail_snapshot is not None else None,
                            detail_reliability=float(best_candidate.detail_snapshot.reliability) if best_candidate.detail_snapshot is not None else None,
                        )

                        track_to_person[detection.track_id] = person_id
                        track_to_last_score[detection.track_id] = score
                        track_is_tentative[detection.track_id] = event_type == "pending_weak_match"
                        recent_person_positions[person_id] = (int(best_candidate.frame_index), self._bbox_center(best_candidate.bbox_xyxy))
                        track_candidates[detection.track_id] = []

                    elif quality_score >= self.config.min_update_quality and not track_is_tentative.get(detection.track_id, False):
                        snapshot_path = save_crop(candidate.crop, self.paths.snapshot_dir, person_id, frame_index)
                        self.store.add_or_update_person(
                            person_id=person_id,
                            embedding=normalize_vector(candidate.embedding),
                            source=self._source_to_label(source),
                            frame_index=frame_index,
                            track_id=detection.track_id,
                            bbox_xyxy=detection.bbox_xyxy,
                            score=score,
                            snapshot_path=str(snapshot_path),
                            payload=self._quality_payload(
                                run_id=run_id,
                                event_type="quality_gated_update",
                                candidate=candidate,
                                good_frame_count=1,
                                motion=motion,
                            ),
                        )
                        recent_person_positions[person_id] = (int(frame_index), self._bbox_center(detection.bbox_xyxy))
                        emit_pipeline_event(
                            "quality_gated_update",
                            should_reid=True,
                            quality_score=float(quality_score),
                            quality_details={key: float(value) for key, value in quality_details.items()},
                            good_frame_count=1,
                            created_new_person=False,
                            detail_label=detail_snapshot.label if detail_snapshot is not None else None,
                            detail_reliability=float(detail_snapshot.reliability) if detail_snapshot is not None else None,
                        )
                    else:
                        if not track_is_tentative.get(detection.track_id, False):
                            skipped_low_quality += 1
                        emit_pipeline_event(
                            "low_quality_update_skipped",
                            should_reid=True,
                            skip_reason="tentative_match_not_updated" if track_is_tentative.get(detection.track_id, False) else "quality_below_min_update",
                            quality_score=float(quality_score),
                            quality_details={key: float(value) for key, value in quality_details.items()},
                            created_new_person=False,
                            detail_label=detail_snapshot.label if detail_snapshot is not None else None,
                            detail_reliability=float(detail_snapshot.reliability) if detail_snapshot is not None else None,
                        )

                else:
                    emit_pipeline_event(
                        "tracking_only",
                        should_reid=False,
                        created_new_person=False,
                    )

                if person_id:
                    recent_person_positions[person_id] = (int(frame_index), self._bbox_center(detection.bbox_xyxy))
                    used_person_ids_this_frame.add(person_id)
                    assigned_person_boxes_this_frame.append((person_id, detection.bbox_xyxy))

                if self.config.draw_debug:
                    draw_detection(
                        frame,
                        detection,
                        person_id,
                        score,
                        motion=motion,
                        draw_motion=self.config.draw_motion_vectors and internal_motion_analysis_enabled,
                        detail_label=detail_label if self.config.draw_detail_labels else None,
                    )

            writer.write(frame)

            if frame_callback and (
                frame_index == 1 or frame_index % max(1, self.config.live_preview_every_n_frames) == 0
            ):
                frame_callback(frame_index, frame)

            if progress_callback and (frame_index == 1 or frame_index % 10 == 0):
                progress_callback(
                    frame_index,
                    total_for_progress,
                    (
                        f"[{self.config.mode_id}] Processed frame {frame_index} | "
                        f"detections: {len(detections)} | skipped low quality: {skipped_low_quality} | details: {detail_observations}"
                    ),
                )

        cap.release()
        writer.release()

        if progress_callback:
            progress_callback(frame_index, total_for_progress, f"[{self.config.mode_id}] Finished")

        if skipped_low_quality > 0:
            warnings.append(f"Skipped low-quality ReID crops: {skipped_low_quality}")
        if waiting_for_good_frames > 0:
            warnings.append(
                "Some tracks were not assigned immediately because the pipeline waited for enough good frames."
            )
        if pending_weak_match_events > 0:
            warnings.append(
                f"Pending weak ReID assignments: {pending_weak_match_events}. "
                "These assignments are shown for review but do not update stored person profiles."
            )
        if overlap_suppressed_events > 0:
            warnings.append(
                f"Delayed new-person evidence for {overlap_suppressed_events} strongly overlapping detections "
                "to avoid duplicate identities from parallel tracker boxes."
            )
        if large_motion_jumps > 0:
            warnings.append(
                f"Motion analysis detected {large_motion_jumps} large track jumps. "
                "Check these scenes for ID switches or camera cuts."
            )
        if self.config.enable_detail_analysis and detail_observations == 0:
            warnings.append(
                "Detail analysis was enabled, but no usable detail snapshots were extracted. "
                "Use larger, sharper person crops or lower the detail confidence threshold for testing."
            )

        persons = self.store.list_persons()
        return PipelineResult(
            output_video_path=output_path,
            output_video_codec=output_codec,
            processed_frames=frame_index,
            created_persons=created_persons,
            matched_events=matched_events,
            strong_match_events=strong_match_events,
            pending_weak_match_events=pending_weak_match_events,
            persons=persons,
            warnings=warnings,
            run_id=run_id,
            mode_id=self.config.mode_id,
            mode_name=self.config.mode_name,
            pipeline_type=self.config.pipeline_type,
        )
