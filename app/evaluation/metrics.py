"""Metrics for controlled single-person videos from ``Details_Tracking``.

These metrics deliberately do not claim multi-person IDF1/MOTA.  They are for
videos known to contain one real target and make fragmentation and continuity
visible without dense bounding-box ground truth.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


def _not_empty(value: Any) -> bool:
    return value is not None and value != "" and str(value).lower() != "nan"


def _dominant(values: list[Any]) -> tuple[Any | None, float, dict[str, int]]:
    clean = [value for value in values if _not_empty(value)]
    if not clean:
        return None, 0.0, {}
    counts = Counter(clean)
    dominant, count = counts.most_common(1)[0]
    return dominant, count / len(clean), {str(key): int(value) for key, value in counts.items()}


def _switch_count(events: list[dict[str, Any]], key: str) -> int:
    previous: Any | None = None
    switches = 0
    for event in sorted(events, key=lambda item: int(item.get("frame_index") or 0)):
        value = event.get(key)
        if not _not_empty(value):
            continue
        if previous is not None and value != previous:
            switches += 1
        previous = value
    return switches


def _ratio(count: int, total: int) -> float | None:
    return None if total <= 0 else count / total


@dataclass(frozen=True)
class RatingThresholds:
    good: float = .90
    medium: float = .75


def rate_ratio(value: float | None, thresholds: RatingThresholds = RatingThresholds()) -> str:
    if value is None:
        return "nicht_bewertbar"
    if value >= thresholds.good:
        return "gut"
    if value >= thresholds.medium:
        return "mittel"
    return "schlecht"


def compute_single_person_metrics(events: list[dict[str, Any]], *, expected_person_id: str | None = None,
                                  processed_frames: int | None = None,
                                  expected_real_person_count: int = 1,
                                  new_person_ids_from_db: list[str] | None = None) -> dict[str, Any]:
    expected_person_id = expected_person_id or None
    new_person_ids_from_db = new_person_ids_from_db or []
    with_track = [event for event in events if _not_empty(event.get("track_id"))]
    with_person = [event for event in events if _not_empty(event.get("person_id"))]
    track_values = [event["track_id"] for event in with_track]
    person_values = [event["person_id"] for event in with_person]
    dominant_track, dominant_track_ratio, track_counts = _dominant(track_values)
    dominant_person, dominant_person_ratio, person_counts = _dominant(person_values)
    frame_indices = {int(event["frame_index"]) for event in events if _not_empty(event.get("frame_index"))}
    expected_events = sum(1 for event in with_person if event.get("person_id") == expected_person_id) if expected_person_id else 0
    expected_ratio = _ratio(expected_events, len(with_person)) if expected_person_id else None
    primary_person = expected_person_id or (str(dominant_person) if dominant_person is not None else None)
    primary_events = sum(1 for event in with_person if event.get("person_id") == primary_person) if primary_person else 0
    primary_ratio = _ratio(primary_events, len(with_person))
    event_type_counts = Counter(str(event.get("event_type") or "unknown") for event in events)
    zone_counts = Counter(str(event.get("decision_zone") or "unknown") for event in events if _not_empty(event.get("decision_zone")))
    unique_person_count = len({str(value) for value in person_values})
    growth_events = [event for event in with_person if event.get("person_id") == primary_person
                     and event.get("event_type") in {"matched_person", "quality_gated_update", "profile_update"}]

    metrics: dict[str, Any] = {
        "expected_real_person_count": int(expected_real_person_count),
        "expected_person_id": expected_person_id,
        "processed_frames": int(processed_frames or 0),
        "event_count": len(events),
        "frames_with_events": len(frame_indices),
        "event_frame_ratio": _ratio(len(frame_indices), int(processed_frames or 0)),
        "track_event_count": len(with_track),
        "unique_track_ids": len({str(value) for value in track_values}),
        "dominant_track_id": str(dominant_track) if dominant_track is not None else None,
        "dominant_track_ratio": dominant_track_ratio,
        "track_switch_count": _switch_count(with_track, "track_id"),
        "track_counts": track_counts,
        "person_event_count": len(with_person),
        "unique_person_ids": unique_person_count,
        "dominant_person_id": str(dominant_person) if dominant_person is not None else None,
        "dominant_person_ratio": dominant_person_ratio,
        "person_switch_count": _switch_count(with_person, "person_id"),
        "person_counts": person_counts,
        "expected_person_events": expected_events,
        "expected_person_ratio": expected_ratio,
        "primary_person_id": primary_person,
        "primary_person_events": primary_events,
        "primary_person_ratio": primary_ratio,
        "profile_growth_event_count": len(growth_events),
        "profile_fragmentation_index": max(0, unique_person_count - expected_real_person_count),
        "created_person_event_count": int(event_type_counts.get("new_person", 0)),
        "new_person_ids_from_db": sorted(new_person_ids_from_db),
        "new_person_count_from_db": len(new_person_ids_from_db),
        "event_type_counts": dict(event_type_counts),
        "decision_zone_counts": dict(zone_counts),
        "pending_weak_match_count": int(event_type_counts.get("pending_weak_match", 0)),
        "pending_new_person_count": int(event_type_counts.get("pending_new_person", 0)),
        "strong_match_count": int(zone_counts.get("strong", 0)),
        "weak_match_count": int(zone_counts.get("weak", 0)),
        "low_match_count": int(zone_counts.get("low", 0)),
    }
    metrics["tracking_rating"] = rate_ratio(dominant_track_ratio if with_track else None)
    metrics["reid_rating"] = rate_ratio(expected_ratio if expected_person_id else dominant_person_ratio)
    metrics["adaptive_reid_rating"] = rate_ratio(primary_ratio)
    metrics["fragmentation_rating"] = (
        "gut" if unique_person_count <= expected_real_person_count
        else "mittel" if unique_person_count <= expected_real_person_count + 1 else "schlecht"
    )
    return metrics


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f} %"


def summary_markdown(*, video_name: str, condition: str, metrics: dict[str, Any], notes: str = "") -> str:
    return f"""# Einzelpersonen-Evaluation – {video_name}

- Bedingung: `{condition or 'nicht gesetzt'}`
- Notizen: {notes or '-'}
- Erwartete Person-ID: `{metrics.get('expected_person_id') or 'nicht gesetzt'}`
- Primäres Profil: `{metrics.get('primary_person_id') or 'nicht gesetzt'}`

| Metrik | Ergebnis |
|---|---:|
| Verarbeitete Frames | {metrics.get('processed_frames')} |
| Unique Track-IDs | {metrics.get('unique_track_ids')} |
| Dominant Track Ratio | {_pct(metrics.get('dominant_track_ratio'))} |
| Track-Wechsel | {metrics.get('track_switch_count')} |
| Unique Person-IDs | {metrics.get('unique_person_ids')} |
| Dominant Person Ratio | {_pct(metrics.get('dominant_person_ratio'))} |
| Expected Person Ratio | {_pct(metrics.get('expected_person_ratio'))} |
| Person-Wechsel | {metrics.get('person_switch_count')} |
| Fragmentierungsindex | {metrics.get('profile_fragmentation_index')} |
| Strong / Weak / Low | {metrics.get('strong_match_count')} / {metrics.get('weak_match_count')} / {metrics.get('low_match_count')} |
| Tracking-Bewertung | **{metrics.get('tracking_rating')}** |
| ReID-Bewertung | **{metrics.get('reid_rating')}** |
| Fragmentierung | **{metrics.get('fragmentation_rating')}** |

Diese Kennzahlen gelten für kontrollierte Einzelpersonen-Videos. Sie sind keine
Multi-Person-IDF1/MOTA-Auswertung ohne dichte Ground-Truth-Trajektorien.
"""
