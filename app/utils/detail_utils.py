"""Explainable, lightweight appearance details for optional ReID re-ranking.

The signals in this module are deliberately weak computer-vision heuristics.
They never identify a person on their own.  A neutral detail score is ``0.5``;
the caller may use the deviation from that value as a small correction to the
primary embedding similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np


DetailKind = Literal["binary", "vector"]


@dataclass(frozen=True)
class DetailSpec:
    name: str
    label: str
    group: str
    kind: DetailKind
    weight: float
    volatile: bool
    length: int
    description: str


DETAIL_REGISTRY: tuple[DetailSpec, ...] = (
    DetailSpec("glasses", "Brille", "head", "binary", .04, True, 2, "Kanten und Dunkelheit im Augenband"),
    DetailSpec("cap", "Kappe/Mütze", "head", "binary", .03, True, 2, "Kontrast in der oberen Kopfregion"),
    DetailSpec("hood", "Kapuze", "head", "binary", .03, True, 2, "Seitlicher Kopf-/Schulterkontrast"),
    DetailSpec("long_hair", "Längere Haare", "head", "binary", .04, False, 2, "Dunkle seitliche Kopf-/Nackenregionen"),
    DetailSpec("beard", "Gesichtsbehaarung", "head", "binary", .04, False, 2, "Textur und Dunkelheit im unteren Gesicht"),
    DetailSpec("watch", "Uhr", "accessory", "binary", .015, True, 2, "Schwaches Signal in den Handgelenkzonen"),
    DetailSpec("backpack", "Rucksack/Tasche", "accessory", "binary", .04, True, 2, "Seitlicher Schulterkontrast"),
    DetailSpec("upper_color_hist", "Oberkörperfarbe", "torso", "vector", .16, False, 18, "HSV-Histogramm des Oberkörpers"),
    DetailSpec("lower_color_hist", "Unterkörperfarbe", "lower_body", "vector", .10, False, 18, "HSV-Histogramm des Unterkörpers"),
    DetailSpec("head_color_hist", "Kopf-/Haarregion", "head", "vector", .06, True, 18, "HSV-Histogramm der Kopfregion"),
    DetailSpec("shirt_print_texture", "Shirt-Aufdruck/Muster", "torso", "vector", .08, True, 6, "Texturprofil der Oberkörpermitte"),
    DetailSpec("torso_region_profile", "Oberkörperstruktur", "torso", "vector", .12, False, 12, "Kompaktes Strukturprofil des Oberkörpers"),
)
DETAIL_SPEC_BY_NAME = {spec.name: spec for spec in DETAIL_REGISTRY}
DETAIL_VECTOR_LENGTH = sum(spec.length for spec in DETAIL_REGISTRY)


@dataclass(frozen=True)
class DetailFeature:
    name: str
    status: str
    confidence: float
    values: tuple[float, ...]

    def to_payload(self) -> dict[str, Any]:
        spec = DETAIL_SPEC_BY_NAME[self.name]
        return {
            "name": self.name,
            "label": spec.label,
            "group": spec.group,
            "kind": spec.kind,
            "weight": spec.weight,
            "volatile": spec.volatile,
            "status": self.status,
            "confidence": float(self.confidence),
            "values": list(self.values),
            "description": spec.description,
        }


@dataclass(frozen=True)
class DetailSnapshot:
    features: dict[str, DetailFeature]
    vector: tuple[float, ...]
    label: str
    reliability: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "registry_version": 2,
            "label": self.label,
            "reliability": float(self.reliability),
            "vector": list(self.vector),
            "features": {name: feature.to_payload() for name, feature in self.features.items()},
        }


def registry_groups_payload() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for spec in DETAIL_REGISTRY:
        groups.setdefault(spec.group, []).append({
            "name": spec.name, "label": spec.label, "kind": spec.kind,
            "weight": spec.weight, "volatile": spec.volatile,
            "length": spec.length, "description": spec.description,
        })
    return groups


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _region(crop: np.ndarray, y1: float, y2: float, x1: float = 0.0, x2: float = 1.0) -> np.ndarray | None:
    if crop is None or crop.size == 0:
        return None
    height, width = crop.shape[:2]
    bounds = (
        max(0, min(height, int(round(height * y1)))),
        max(0, min(height, int(round(height * y2)))),
        max(0, min(width, int(round(width * x1)))),
        max(0, min(width, int(round(width * x2)))),
    )
    yy1, yy2, xx1, xx2 = bounds
    if yy2 <= yy1 or xx2 <= xx1:
        return None
    result = crop[yy1:yy2, xx1:xx2]
    return result if result.size else None


def _gray(region: np.ndarray | None) -> np.ndarray | None:
    if region is None or region.size == 0:
        return None
    return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region


def _edge_density(region: np.ndarray | None) -> float:
    gray = _gray(region)
    if gray is None or min(gray.shape[:2]) < 4:
        return 0.0
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    return _clip01(float(np.count_nonzero(edges)) / float(edges.size) * 8.0)


def _texture(region: np.ndarray | None) -> float:
    gray = _gray(region)
    if gray is None or min(gray.shape[:2]) < 4:
        return 0.0
    return _clip01(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 900.0)


def _dark(region: np.ndarray | None, threshold: int = 85) -> float:
    gray = _gray(region)
    return 0.0 if gray is None else _clip01(float(np.mean(gray < threshold)) * 2.0)


def _bright(region: np.ndarray | None, threshold: int = 185) -> float:
    gray = _gray(region)
    return 0.0 if gray is None else _clip01(float(np.mean(gray > threshold)) * 1.5)


def _contrast(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None or not first.size or not second.size:
        return 0.0
    first_mean = np.mean(first.reshape(-1, first.shape[-1]), axis=0)
    second_mean = np.mean(second.reshape(-1, second.shape[-1]), axis=0)
    return _clip01(float(np.linalg.norm(first_mean - second_mean)) / 90.0)


def _hist(region: np.ndarray | None) -> tuple[float, ...]:
    if region is None or not region.size or region.ndim != 3:
        return (0.0,) * 18
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    values = cv2.calcHist([hsv], [0, 1], None, [6, 3], [0, 180, 0, 256]).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    return tuple(float(item) for item in values / norm) if norm > 1e-6 else (0.0,) * 18


def _binary(name: str, present: float, absent: float, minimum: float) -> DetailFeature:
    present, absent = _clip01(present), _clip01(absent)
    confidence = max(present, absent)
    status = "unknown" if confidence < minimum else "yes" if present >= absent else "no"
    return DetailFeature(name, status, confidence, (present, absent))


def _vector(name: str, values: tuple[float, ...], confidence: float) -> DetailFeature:
    length = DETAIL_SPEC_BY_NAME[name].length
    cleaned = tuple(_clip01(value) for value in values[:length])
    cleaned += (0.0,) * (length - len(cleaned))
    return DetailFeature(name, "observed", _clip01(confidence), cleaned)


def _small_profile(region: np.ndarray | None) -> tuple[float, ...]:
    if region is None:
        return (0.0,) * 12
    values: list[float] = []
    for y1, y2 in ((0.0, .33), (.33, .66), (.66, 1.0)):
        band = _region(region, y1, y2)
        gray = _gray(band)
        if gray is None:
            values.extend((0.0,) * 4)
        else:
            values.extend((_clip01(float(np.mean(gray)) / 255), _clip01(float(np.std(gray)) / 90),
                           _edge_density(band), _texture(band)))
    return tuple(values)


def _shirt_texture(region: np.ndarray | None) -> tuple[float, ...]:
    if region is None:
        return (0.0,) * 6
    center = _region(region, .18, .82, .2, .8)
    left, right = _region(region, .18, .82, 0, .25), _region(region, .18, .82, .75, 1)
    upper, lower = _region(region, 0, .35, .2, .8), _region(region, .65, 1, .2, .8)
    return (_edge_density(center), _texture(center), _bright(center), _dark(center, 70),
            _contrast(center, left), .5 * _contrast(center, right) + .5 * _contrast(upper, lower))


def extract_detail_snapshot(crop: np.ndarray | None, min_confidence: float = .55) -> DetailSnapshot | None:
    """Extract the Details_Tracking registry from one sufficiently large crop."""
    if crop is None or crop.size == 0:
        return None
    height, width = crop.shape[:2]
    if width < 24 or height < 72:
        return None

    eye, forehead = _region(crop, .10, .24, .18, .82), _region(crop, .03, .10, .22, .78)
    top, face = _region(crop, 0, .13, .15, .85), _region(crop, .14, .30, .18, .82)
    left_head, right_head = _region(crop, .06, .33, .02, .24), _region(crop, .06, .33, .76, .98)
    face_center, shoulders = _region(crop, .12, .32, .32, .68), _region(crop, .30, .52, .18, .82)
    lower_face, upper_face = _region(crop, .22, .34, .30, .70), _region(crop, .13, .21, .30, .70)
    left_wrist, right_wrist = _region(crop, .48, .86, 0, .22), _region(crop, .48, .86, .78, 1)
    torso_center = _region(crop, .28, .75, .25, .75)
    left_side, right_side = _region(crop, .25, .75, 0, .20), _region(crop, .25, .75, .80, 1)
    head, upper, lower = _region(crop, 0, .34, .12, .88), _region(crop, .28, .66, .08, .92), _region(crop, .62, 1, .10, .90)

    def binary_signal(name: str, signal: float) -> DetailFeature:
        return _binary(name, signal, 1.0 - signal, min_confidence)

    glasses = .48 * _edge_density(eye) + .32 * _dark(eye) + .20 * _contrast(eye, forehead)
    cap = .45 * _contrast(top, face) + .35 * _dark(top, 95) + .20 * _edge_density(top)
    hood = .40 * max(_contrast(left_head, face_center), _contrast(right_head, face_center)) + .35 * _contrast(shoulders, face_center) + .25 * max(_edge_density(left_head), _edge_density(right_head))
    long_hair = .40 * max(_dark(left_head, 95), _dark(right_head, 95)) + .35 * max(_contrast(left_head, face_center), _contrast(right_head, face_center)) + .25 * max(_contrast(left_head, shoulders), _contrast(right_head, shoulders))
    beard = .42 * _dark(lower_face, 105) + .30 * _texture(lower_face) + .28 * _contrast(lower_face, upper_face)
    watch = .60 * max(.55 * _edge_density(left_wrist) + .45 * _dark(left_wrist, 75), .55 * _edge_density(right_wrist) + .45 * _dark(right_wrist, 75)) + .40 * max(_contrast(left_wrist, torso_center), _contrast(right_wrist, torso_center))
    backpack = .45 * max(_contrast(left_side, torso_center), _contrast(right_side, torso_center)) + .30 * max(_edge_density(left_side), _edge_density(right_side)) + .25 * max(_dark(left_side, 90), _dark(right_side, 90))

    features = {
        "glasses": binary_signal("glasses", glasses), "cap": binary_signal("cap", cap),
        "hood": binary_signal("hood", hood), "long_hair": binary_signal("long_hair", long_hair),
        "beard": binary_signal("beard", beard), "watch": binary_signal("watch", watch),
        "backpack": binary_signal("backpack", backpack),
        "upper_color_hist": _vector("upper_color_hist", _hist(upper), .85),
        "lower_color_hist": _vector("lower_color_hist", _hist(lower), .80),
        "head_color_hist": _vector("head_color_hist", _hist(head), .65),
        "shirt_print_texture": _vector("shirt_print_texture", _shirt_texture(upper), .65),
        "torso_region_profile": _vector("torso_region_profile", _small_profile(upper), .75),
    }
    vector = tuple(value for spec in DETAIL_REGISTRY for value in features[spec.name].values)
    reliability = float(np.mean([feature.confidence for feature in features.values()]))
    labels = [f"{name}:{features[name].status}" for name in ("glasses", "cap", "hood", "long_hair", "beard", "backpack")
              if features[name].status != "unknown"]
    return DetailSnapshot(features, vector, ", ".join(labels[:4]) or "details:regions", _clip01(reliability))


def detail_similarity_breakdown(query_vector: list[float] | tuple[float, ...] | np.ndarray | None,
                                stored_vector: list[float] | tuple[float, ...] | np.ndarray | None) -> dict[str, Any]:
    if query_vector is None or stored_vector is None:
        return {"score": .5, "reason": "missing_detail_vector", "features": []}
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    stored = np.asarray(stored_vector, dtype=np.float32).reshape(-1)
    if query.shape != stored.shape or query.size != DETAIL_VECTOR_LENGTH:
        return {"score": .5, "reason": "incompatible_detail_vector", "features": []}

    offset, influence_sum, weight_sum = 0, 0.0, 0.0
    rows: list[dict[str, Any]] = []
    for spec in DETAIL_REGISTRY:
        q_part, s_part = query[offset:offset + spec.length], stored[offset:offset + spec.length]
        offset += spec.length
        q_norm, s_norm = float(np.linalg.norm(q_part)), float(np.linalg.norm(s_part))
        ignored = q_norm < 1e-6 or s_norm < 1e-6
        if spec.kind == "binary" and (float(np.max(q_part)) < .45 or float(np.max(s_part)) < .45):
            ignored = True
        score = .5 if ignored else _clip01((float(np.dot(q_part / q_norm, s_part / s_norm)) + 1) / 2)
        effective = 0.0 if ignored else spec.weight
        if score < .5:
            effective *= .25 if spec.volatile else .5
        influence = (score - .5) * effective
        influence_sum, weight_sum = influence_sum + influence, weight_sum + effective
        rows.append({"name": spec.name, "label": spec.label, "group": spec.group, "score": score,
                     "weight": spec.weight, "effective_weight": effective, "volatile": spec.volatile,
                     "influence": influence, "ignored": ignored})
    rows.sort(key=lambda row: abs(float(row["influence"])), reverse=True)
    if weight_sum <= 1e-6:
        return {"score": .5, "reason": "all_details_neutral_or_unknown", "features": rows}
    return {"score": _clip01(.5 + influence_sum / weight_sum), "reason": "ok_conservative_details", "features": rows}


def combine_visual_and_detail_score(visual_score: float, detail_score: float | None, detail_weight: float) -> float:
    """Keep cosine similarity primary and apply only a bounded detail correction."""
    if detail_score is None:
        return float(np.clip(visual_score, -1.0, 1.0))
    corrected = float(visual_score) + (_clip01(detail_score) - .5) * _clip01(detail_weight)
    return float(np.clip(corrected, -1.0, 1.0))


def compact_detail_influences(breakdown: dict[str, Any] | None, limit: int = 5) -> str:
    if not breakdown or not isinstance(breakdown.get("features"), list):
        return ""
    parts = []
    for item in breakdown["features"][:limit]:
        try:
            parts.append(f"{item.get('label', item.get('name'))}: {float(item['score']):.2f} ({float(item['influence']):+.3f})")
        except (KeyError, TypeError, ValueError):
            continue
    return "; ".join(parts)
