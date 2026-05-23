from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from app.config import AppPaths, PipelineConfig
from app.pipeline.detector_tracker import UltralyticsPersonTracker
from app.pipeline.reid_encoder import build_encoder
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
    save_crop,
)

ProgressCallback = Callable[[int, int | None, str], None]
FrameCallback = Callable[[int, np.ndarray], None]


@dataclass
class TrackEmbeddingCandidate:
    embedding: np.ndarray
    quality_score: float
    quality_details: dict[str, float]
    crop: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    frame_index: int
    detection_confidence: float


class PersonReIdPipeline:
    def __init__(self, config: PipelineConfig, paths: AppPaths | None = None) -> None:
        self.config = config
        self.paths = paths or AppPaths()
        self.paths.ensure()
        self.store = SQLiteVectorStore(self.paths.db_path)
        self.tracker = UltralyticsPersonTracker(
            model_name=config.yolo_model,
            tracker=config.tracker,
            confidence=config.detection_confidence,
            image_size=config.image_size,
            device=config.device,
        )
        self.encoder = build_encoder(config.encoder_backend, device=config.device)

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

    def _quality_payload(
        self,
        *,
        run_id: str,
        event_type: str,
        candidate: TrackEmbeddingCandidate,
        good_frame_count: int,
        quality_average: float | None = None,
    ) -> dict[str, object]:
        quality_average = candidate.quality_score if quality_average is None else quality_average
        return {
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

    def process(
        self,
        source: str | int | CameraSource,
        progress_callback: ProgressCallback | None = None,
        frame_callback: FrameCallback | None = None,
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
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

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
                "encoder_backend": self.config.encoder_backend,
                "match_threshold": self.config.match_threshold,
                "detection_confidence": self.config.detection_confidence,
                "image_size": self.config.image_size,
                "reid_every_n_frames": self.config.reid_every_n_frames,
                "min_good_frames_before_reid": self.config.min_good_frames_before_reid,
                "min_embedding_quality": self.config.min_embedding_quality,
                "min_update_quality": self.config.min_update_quality,
                "enable_ball_tracking": self.config.enable_ball_tracking,
                "enable_pitch_mapping": self.config.enable_pitch_mapping,
                "enable_team_classification": self.config.enable_team_classification,
                "enable_stats_aggregation": self.config.enable_stats_aggregation,
            },
        )

        track_to_person: dict[int, str] = {}
        track_to_last_score: dict[int, float | None] = {}
        track_candidates: dict[int, list[TrackEmbeddingCandidate]] = {}
        created_persons = 0
        matched_events = 0
        frame_index = 0
        skipped_low_quality = 0
        waiting_for_good_frames = 0
        warnings: list[str] = []
        mode_warning = self._mode_warning()
        if mode_warning:
            warnings.append(mode_warning)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if self.config.max_frames > 0 and frame_index > self.config.max_frames:
                break

            detections = self.tracker.track_frame(frame)
            used_person_ids_this_frame: set[str] = set()

            for detection in detections:
                person_id = track_to_person.get(detection.track_id)
                score = track_to_last_score.get(detection.track_id)
                should_reid = person_id is None or frame_index % max(1, self.config.reid_every_n_frames) == 0

                if should_reid:
                    crop = crop_xyxy(frame, detection.bbox_xyxy, padding=self.config.crop_padding)
                    if not is_valid_crop(crop, self.config.min_crop_width, self.config.min_crop_height):
                        skipped_low_quality += 1
                        if self.config.draw_debug:
                            draw_detection(frame, detection, person_id, score)
                        continue

                    quality_score, quality_details = crop_quality_score(
                        crop=crop,
                        bbox_xyxy=detection.bbox_xyxy,
                        frame_shape=frame.shape,
                        detection_confidence=detection.confidence,
                        min_width=self.config.min_crop_width,
                        min_height=self.config.min_crop_height,
                    )
                    if quality_score < self.config.min_embedding_quality:
                        skipped_low_quality += 1
                        if self.config.draw_debug:
                            draw_detection(frame, detection, person_id, score)
                        continue

                    embedding = self.encoder.encode(crop)
                    candidate = TrackEmbeddingCandidate(
                        embedding=embedding,
                        quality_score=quality_score,
                        quality_details=quality_details,
                        crop=crop.copy(),
                        bbox_xyxy=detection.bbox_xyxy,
                        frame_index=frame_index,
                        detection_confidence=detection.confidence,
                    )

                    if person_id is None:
                        candidates = track_candidates.setdefault(detection.track_id, [])
                        candidates.append(candidate)
                        max_buffer_size = max(self.config.min_good_frames_before_reid * 2, 5)
                        if len(candidates) > max_buffer_size:
                            del candidates[0 : len(candidates) - max_buffer_size]

                        if len(candidates) < self.config.min_good_frames_before_reid:
                            waiting_for_good_frames += 1
                            if self.config.draw_debug:
                                draw_detection(frame, detection, person_id, score)
                            continue

                        combined_embedding = self._combined_embedding(candidates)
                        best_candidate = self._best_candidate(candidates)
                        quality_average = float(np.mean([item.quality_score for item in candidates]))
                        match = self.store.search(
                            combined_embedding,
                            threshold=self.config.match_threshold,
                            exclude_person_ids=used_person_ids_this_frame,
                        )

                        if match is None:
                            person_id = self.store.create_person_id()
                            score = None
                            created_persons += 1
                        else:
                            person_id = match.person_id
                            score = match.score
                            matched_events += 1

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
                                event_type="initial_buffer_match",
                                candidate=best_candidate,
                                good_frame_count=len(candidates),
                                quality_average=quality_average,
                            ),
                        )

                        track_to_person[detection.track_id] = person_id
                        track_to_last_score[detection.track_id] = score
                        track_candidates[detection.track_id] = []

                    elif quality_score >= self.config.min_update_quality:
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
                            ),
                        )
                    else:
                        skipped_low_quality += 1

                if person_id:
                    used_person_ids_this_frame.add(person_id)

                if self.config.draw_debug:
                    draw_detection(frame, detection, person_id, score)

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
                        f"detections: {len(detections)} | skipped low quality: {skipped_low_quality}"
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

        persons = self.store.list_persons()
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
        )
