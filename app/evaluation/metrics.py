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
    return dominant, float(count / len(clean)), {str(k): int(v) for k, v in counts.items()}


def _switch_count(events: list[dict[str, Any]], key: str) -> int:
    last_value: Any | None = None
    switches = 0
    for event in sorted(events, key=lambda item: int(item.get("frame_index") or 0)):
        value = event.get(key)
        if not _not_empty(value):
            continue
        if last_value is not None and value != last_value:
            switches += 1
        last_value = value
    return switches


def _ratio(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return float(count / total)


@dataclass(frozen=True)
class RatingThresholds:
    good: float = 0.90
    medium: float = 0.75


def rate_ratio(value: float | None, thresholds: RatingThresholds = RatingThresholds()) -> str:
    if value is None:
        return "nicht_bewertbar"
    if value >= thresholds.good:
        return "gut"
    if value >= thresholds.medium:
        return "mittel"
    return "schlecht"


def compute_single_person_metrics(
    events: list[dict[str, Any]],
    *,
    expected_person_id: str | None = None,
    processed_frames: int | None = None,
    expected_real_person_count: int = 1,
    new_person_ids_from_db: list[str] | None = None,
) -> dict[str, Any]:
    """Compute pragmatic single-person ReID evaluation metrics.

    Assumption: each test video contains one real target person. This is not a
    multi-person IDF1/MOTA benchmark. It is designed for controlled videos where
    one known person is seen under different outfit/detail conditions.
    """

    expected_person_id = expected_person_id or None
    new_person_ids_from_db = new_person_ids_from_db or []

    events_with_track = [event for event in events if _not_empty(event.get("track_id"))]
    events_with_person = [event for event in events if _not_empty(event.get("person_id"))]
    reid_decision_events = [
        event
        for event in events_with_person
        if event.get("event_type")
        in {"new_person", "matched_person", "quality_gated_update", "tracking_only"}
    ]

    track_values = [event.get("track_id") for event in events_with_track]
    person_values = [event.get("person_id") for event in events_with_person]

    dominant_track_id, dominant_track_ratio, track_counts = _dominant(track_values)
    dominant_person_id, dominant_person_ratio, person_counts = _dominant(person_values)

    unique_frame_indices = {
        int(event.get("frame_index"))
        for event in events
        if _not_empty(event.get("frame_index"))
    }

    expected_person_events = 0
    expected_person_ratio = None
    if expected_person_id:
        expected_person_events = sum(1 for event in events_with_person if event.get("person_id") == expected_person_id)
        expected_person_ratio = _ratio(expected_person_events, len(events_with_person))

    # Adaptive-calibration interpretation:
    # The calibration ID is a start profile, not a hard single source of truth.
    # If an expected ID exists, it is still the primary profile. Otherwise the
    # dominant person ID is treated as the primary profile for exploratory runs.
    primary_person_id = expected_person_id or (str(dominant_person_id) if dominant_person_id is not None else None)
    primary_person_events = 0
    primary_person_ratio = None
    if primary_person_id:
        primary_person_events = sum(1 for event in events_with_person if event.get("person_id") == primary_person_id)
        primary_person_ratio = _ratio(primary_person_events, len(events_with_person))

    created_person_events = [event for event in events if bool(event.get("created_new_person"))]
    event_type_counts = Counter(str(event.get("event_type") or "unknown") for event in events)
    decision_zone_counts = Counter(str(event.get("decision_zone") or "unknown") for event in events if _not_empty(event.get("decision_zone")))

    profile_growth_events = [
        event
        for event in events_with_person
        if event.get("person_id") == primary_person_id
        and str(event.get("event_type") or "") in {"matched_person", "quality_gated_update"}
    ]

    unique_person_count = int(len(set(str(value) for value in person_values if _not_empty(value))))
    profile_fragmentation_index = max(0, unique_person_count - int(expected_real_person_count))

    metrics: dict[str, Any] = {
        "expected_real_person_count": int(expected_real_person_count),
        "expected_person_id": expected_person_id,
        "processed_frames": int(processed_frames or 0),
        "event_count": int(len(events)),
        "frames_with_events": int(len(unique_frame_indices)),
        "event_frame_ratio": _ratio(len(unique_frame_indices), int(processed_frames or 0)),
        "track_event_count": int(len(events_with_track)),
        "unique_track_ids": int(len(set(str(value) for value in track_values if _not_empty(value)))),
        "dominant_track_id": str(dominant_track_id) if dominant_track_id is not None else None,
        "dominant_track_ratio": dominant_track_ratio,
        "track_switch_count": int(_switch_count(events_with_track, "track_id")),
        "track_counts": track_counts,
        "person_event_count": int(len(events_with_person)),
        "reid_decision_event_count": int(len(reid_decision_events)),
        "unique_person_ids": unique_person_count,
        "dominant_person_id": str(dominant_person_id) if dominant_person_id is not None else None,
        "dominant_person_ratio": dominant_person_ratio,
        "person_switch_count": int(_switch_count(events_with_person, "person_id")),
        "person_counts": person_counts,
        "expected_person_events": int(expected_person_events),
        "expected_person_ratio": expected_person_ratio,
        "primary_person_id": primary_person_id,
        "primary_person_events": int(primary_person_events),
        "primary_person_ratio": primary_person_ratio,
        "profile_growth_event_count": int(len(profile_growth_events)),
        "profile_fragmentation_index": int(profile_fragmentation_index),
        "created_person_event_count": int(len(created_person_events)),
        "new_person_ids_from_db": sorted(str(item) for item in new_person_ids_from_db),
        "new_person_count_from_db": int(len(new_person_ids_from_db)),
        "event_type_counts": {str(k): int(v) for k, v in event_type_counts.items()},
        "decision_zone_counts": {str(k): int(v) for k, v in decision_zone_counts.items()},
        "pending_weak_match_count": int(event_type_counts.get("pending_weak_match", 0)),
        "pending_new_person_count": int(event_type_counts.get("pending_new_person", 0)),
        "strong_match_count": int(decision_zone_counts.get("strong", 0)),
        "weak_match_count": int(decision_zone_counts.get("weak", 0)),
        "low_match_count": int(decision_zone_counts.get("low", 0)),
    }

    tracking_ratio_for_rating = dominant_track_ratio if metrics["track_event_count"] else None
    reid_ratio_for_rating = expected_person_ratio if expected_person_id else dominant_person_ratio

    metrics["tracking_rating"] = rate_ratio(tracking_ratio_for_rating)
    metrics["reid_rating"] = rate_ratio(reid_ratio_for_rating)
    metrics["adaptive_reid_rating"] = rate_ratio(primary_person_ratio)
    metrics["fragmentation_rating"] = (
        "gut"
        if metrics["unique_person_ids"] <= expected_real_person_count
        else "mittel"
        if metrics["unique_person_ids"] <= expected_real_person_count + 1
        else "schlecht"
    )
    return metrics


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f} %"


def summary_markdown(*, video_name: str, condition: str, metrics: dict[str, Any], notes: str = "") -> str:
    return f"""# Evaluation – {video_name}

## Kontext

- Bedingung: `{condition or 'nicht gesetzt'}`
- Notizen: {notes or '-'}
- Erwartete echte Personen: {metrics.get('expected_real_person_count')}
- Erwartete Person-ID: `{metrics.get('expected_person_id') or 'nicht gesetzt'}`
- Primäres Profil: `{metrics.get('primary_person_id') or 'nicht gesetzt'}`

## Events

| Wert | Ergebnis |
|---|---:|
| Verarbeitete Frames | {metrics.get('processed_frames')} |
| Event-Zeilen | {metrics.get('event_count')} |
| Frames mit Event | {metrics.get('frames_with_events')} |
| Event-Frame-Quote | {_pct(metrics.get('event_frame_ratio'))} |

## Tracking-Stabilität

| Wert | Ergebnis |
|---|---:|
| Unique Track-IDs | {metrics.get('unique_track_ids')} |
| Dominante Track-ID | `{metrics.get('dominant_track_id')}` |
| Dominant Track Ratio | {_pct(metrics.get('dominant_track_ratio'))} |
| Track Switches | {metrics.get('track_switch_count')} |
| Bewertung | **{metrics.get('tracking_rating')}** |

## ReID-Wiedererkennung

| Wert | Ergebnis |
|---|---:|
| Unique Person-IDs | {metrics.get('unique_person_ids')} |
| Dominante Person-ID | `{metrics.get('dominant_person_id')}` |
| Dominant Person Ratio | {_pct(metrics.get('dominant_person_ratio'))} |
| Expected Person Ratio | {_pct(metrics.get('expected_person_ratio'))} |
| Primary Person Ratio | {_pct(metrics.get('primary_person_ratio'))} |
| Person Switches | {metrics.get('person_switch_count')} |
| Neue Person-IDs laut DB | {metrics.get('new_person_count_from_db')} |
| Profil-Wachstums-Events | {metrics.get('profile_growth_event_count')} |
| Pending Weak Matches | {metrics.get('pending_weak_match_count')} |
| Pending New-Person Events | {metrics.get('pending_new_person_count')} |
| Strong/Weak/Low Match-Zonen | {metrics.get('strong_match_count')} / {metrics.get('weak_match_count')} / {metrics.get('low_match_count')} |
| Fragmentierungsindex | {metrics.get('profile_fragmentation_index')} |
| Starre ReID-Bewertung | **{metrics.get('reid_rating')}** |
| Adaptive ReID-Bewertung | **{metrics.get('adaptive_reid_rating')}** |
| Fragmentierung | **{metrics.get('fragmentation_rating')}** |

## Interpretation

- Viele Track-IDs oder Track-Switches deuten auf Tracking-Instabilität im Video hin.
- Mehrere Person-IDs bei nur einer echten Person deuten auf ReID-Fragmentierung hin.
- Die Expected Person Ratio ist streng und bewertet, ob exakt die Kalibrier-ID getroffen wurde.
- Die Primary Person Ratio bewertet den praxisnäheren Fall, ob das Hauptprofil konsistent weitergeführt wurde.
- Sinkt die Primary Person Ratio bei Brille/Gelb/Gelb+Brille, ist das ein Hinweis auf Detail- oder Outfit-Bias.
"""
