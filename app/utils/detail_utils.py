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


# Registry of all detail signals that are currently extracted from the person crop.
# The registry is intentionally explicit so later UI/debug output can show why a
# match changed and which features were used. These are MVP CV signals; each item
# can later be replaced by a trained classifier without changing the storage API.
DETAIL_REGISTRY: tuple[DetailSpec, ...] = (
    DetailSpec("glasses", "Brille", "head", "binary", 0.04, True, 2, "yes/no confidence from eye-band edges and darkness"),
    DetailSpec("cap", "Kappe/Mütze", "head", "binary", 0.03, True, 2, "yes/no confidence from top-head contrast"),
    DetailSpec("hood", "Kapuze", "head", "binary", 0.03, True, 2, "yes/no confidence from side/top-head contrast"),
    DetailSpec("long_hair", "längere Haare", "head", "binary", 0.04, False, 2, "yes/no confidence from dark side regions around neck/head"),
    DetailSpec("beard", "Gesichtsbehaarung", "head", "binary", 0.04, False, 2, "yes/no confidence from lower-face darkness and texture"),
    DetailSpec("watch", "Uhr", "accessory", "binary", 0.015, True, 2, "weak wrist-zone signal, low weight without pose estimation"),
    DetailSpec("backpack", "Rucksack/Tasche", "accessory", "binary", 0.04, True, 2, "side-shoulder contrast and edge signal"),
    DetailSpec("upper_color_hist", "Oberkörperfarbe", "torso", "vector", 0.16, False, 18, "HSV color histogram of upper body / shirt / pullover / jacket"),
    DetailSpec("lower_color_hist", "Unterkörperfarbe", "lower_body", "vector", 0.10, False, 18, "HSV color histogram of trousers/skirt/lower body"),
    DetailSpec("head_color_hist", "Kopf-/Haarregion", "head", "vector", 0.06, True, 18, "HSV color histogram of head region"),
    DetailSpec("shirt_print_texture", "T-Shirt-Aufdruck/Muster", "torso", "vector", 0.08, True, 6, "texture/edge summary in central upper-body zone"),
    DetailSpec("torso_region_profile", "Oberkörper-Region", "torso", "vector", 0.12, False, 12, "small region profile for clothing structure"),
)

DETAIL_KEYS: tuple[str, ...] = tuple(spec.name for spec in DETAIL_REGISTRY)
DETAIL_SPEC_BY_NAME: dict[str, DetailSpec] = {spec.name: spec for spec in DETAIL_REGISTRY}

# Stable vector key order for storage. If the registry changes, old stored detail
# vectors with a different length are treated as neutral during matching.
DETAIL_VECTOR_KEYS: tuple[str, ...] = tuple(
    f"{spec.name}_{idx}" if spec.kind == "vector" else key
    for spec in DETAIL_REGISTRY
    for idx, key in enumerate(([f"{spec.name}_yes", f"{spec.name}_no"] if spec.kind == "binary" else [str(i) for i in range(spec.length)]))
)


@dataclass(frozen=True)
class DetailFeature:
    name: str
    label: str
    group: str
    kind: DetailKind
    weight: float
    volatile: bool
    status: str
    confidence: float
    values: list[float]
    present_confidence: float | None = None
    absent_confidence: float | None = None
    description: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "label": self.label,
            "group": self.group,
            "kind": self.kind,
            "weight": float(self.weight),
            "volatile": bool(self.volatile),
            "status": self.status,
            "confidence": float(self.confidence),
            "values": [float(value) for value in self.values],
            "description": self.description,
        }
        if self.present_confidence is not None:
            payload["present_confidence"] = float(self.present_confidence)
        if self.absent_confidence is not None:
            payload["absent_confidence"] = float(self.absent_confidence)
        return payload


@dataclass(frozen=True)
class DetailSnapshot:
    features: dict[str, DetailFeature]
    vector: list[float]
    label: str
    reliability: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "reliability": float(self.reliability),
            "registry_version": 2,
            "vector_keys": list(DETAIL_VECTOR_KEYS),
            "vector": [float(value) for value in self.vector],
            "features": {name: feature.to_payload() for name, feature in self.features.items()},
            "groups": registry_groups_payload(),
        }


def registry_groups_payload() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in DETAIL_REGISTRY:
        grouped.setdefault(spec.group, []).append(
            {
                "name": spec.name,
                "label": spec.label,
                "kind": spec.kind,
                "weight": float(spec.weight),
                "volatile": bool(spec.volatile),
                "length": int(spec.length),
                "description": spec.description,
            }
        )
    return grouped


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _safe_region(crop: np.ndarray, y1: float, y2: float, x1: float = 0.0, x2: float = 1.0) -> np.ndarray | None:
    if crop is None or crop.size == 0:
        return None
    height, width = crop.shape[:2]
    yy1 = int(round(height * y1))
    yy2 = int(round(height * y2))
    xx1 = int(round(width * x1))
    xx2 = int(round(width * x2))
    yy1 = max(0, min(height, yy1))
    yy2 = max(0, min(height, yy2))
    xx1 = max(0, min(width, xx1))
    xx2 = max(0, min(width, xx2))
    if yy2 <= yy1 or xx2 <= xx1:
        return None
    region = crop[yy1:yy2, xx1:xx2]
    return region if region.size else None


def _gray(region: np.ndarray | None) -> np.ndarray | None:
    if region is None or region.size == 0:
        return None
    return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region


def _edge_density(region: np.ndarray | None) -> float:
    gray = _gray(region)
    if gray is None or min(gray.shape[:2]) < 4:
        return 0.0
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return _clip01(float(np.count_nonzero(edges)) / float(edges.size) * 8.0)


def _laplacian_texture(region: np.ndarray | None) -> float:
    gray = _gray(region)
    if gray is None or min(gray.shape[:2]) < 4:
        return 0.0
    return _clip01(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 900.0)


def _dark_ratio(region: np.ndarray | None, threshold: int = 80) -> float:
    gray = _gray(region)
    if gray is None:
        return 0.0
    return _clip01(float(np.mean(gray < threshold)) * 2.0)


def _bright_ratio(region: np.ndarray | None, threshold: int = 185) -> float:
    gray = _gray(region)
    if gray is None:
        return 0.0
    return _clip01(float(np.mean(gray > threshold)) * 1.5)


def _color_contrast(region_a: np.ndarray | None, region_b: np.ndarray | None) -> float:
    if region_a is None or region_b is None or region_a.size == 0 or region_b.size == 0:
        return 0.0
    mean_a = np.mean(region_a.reshape(-1, region_a.shape[-1]), axis=0)
    mean_b = np.mean(region_b.reshape(-1, region_b.shape[-1]), axis=0)
    distance = float(np.linalg.norm(mean_a - mean_b))
    return _clip01(distance / 90.0)


def _color_hist(region: np.ndarray | None, h_bins: int = 6, s_bins: int = 3) -> list[float]:
    length = h_bins * s_bins
    if region is None or region.size == 0 or region.ndim != 3:
        return [0.0] * length
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(hist))
    if norm <= 1e-6:
        return [0.0] * length
    return (hist / norm).astype(float).tolist()


def _small_region_profile(region: np.ndarray | None) -> list[float]:
    # 3 horizontal bands x 4 metrics. Compact and cheap regional descriptor.
    if region is None or region.size == 0:
        return [0.0] * 12
    values: list[float] = []
    for y1, y2 in ((0.0, 0.33), (0.33, 0.66), (0.66, 1.0)):
        band = _safe_region(region, y1, y2)
        gray = _gray(band)
        if gray is None:
            values.extend([0.0, 0.0, 0.0, 0.0])
            continue
        values.append(_clip01(float(np.mean(gray)) / 255.0))
        values.append(_clip01(float(np.std(gray)) / 90.0))
        values.append(_edge_density(band))
        values.append(_laplacian_texture(band))
    return [float(_clip01(value)) for value in values]


def _shirt_print_texture(region: np.ndarray | None) -> list[float]:
    center = _safe_region(region, 0.18, 0.82, 0.20, 0.80) if region is not None else None
    left = _safe_region(region, 0.18, 0.82, 0.00, 0.25) if region is not None else None
    right = _safe_region(region, 0.18, 0.82, 0.75, 1.00) if region is not None else None
    upper = _safe_region(region, 0.00, 0.35, 0.20, 0.80) if region is not None else None
    lower = _safe_region(region, 0.65, 1.00, 0.20, 0.80) if region is not None else None
    return [
        _edge_density(center),
        _laplacian_texture(center),
        _bright_ratio(center),
        _dark_ratio(center, threshold=70),
        _color_contrast(center, left),
        _color_contrast(center, right) * 0.5 + _color_contrast(upper, lower) * 0.5,
    ]


def _status(present_confidence: float, absent_confidence: float, min_confidence: float) -> str:
    present_confidence = _clip01(present_confidence)
    absent_confidence = _clip01(absent_confidence)
    if max(present_confidence, absent_confidence) < min_confidence:
        return "unknown"
    return "yes" if present_confidence >= absent_confidence else "no"


def _binary_feature(spec_name: str, present_confidence: float, absent_confidence: float, min_confidence: float) -> DetailFeature:
    spec = DETAIL_SPEC_BY_NAME[spec_name]
    present_confidence = _clip01(present_confidence)
    absent_confidence = _clip01(absent_confidence)
    status = _status(present_confidence, absent_confidence, min_confidence)
    confidence = max(present_confidence, absent_confidence)
    return DetailFeature(
        name=spec.name,
        label=spec.label,
        group=spec.group,
        kind=spec.kind,
        weight=spec.weight,
        volatile=spec.volatile,
        status=status,
        confidence=confidence,
        values=[present_confidence, absent_confidence],
        present_confidence=present_confidence,
        absent_confidence=absent_confidence,
        description=spec.description,
    )


def _vector_feature(spec_name: str, values: list[float], confidence: float, status: str = "observed") -> DetailFeature:
    spec = DETAIL_SPEC_BY_NAME[spec_name]
    cleaned = [float(_clip01(value)) for value in values[: spec.length]]
    if len(cleaned) < spec.length:
        cleaned.extend([0.0] * (spec.length - len(cleaned)))
    return DetailFeature(
        name=spec.name,
        label=spec.label,
        group=spec.group,
        kind=spec.kind,
        weight=spec.weight,
        volatile=spec.volatile,
        status=status,
        confidence=_clip01(confidence),
        values=cleaned,
        description=spec.description,
    )


def _glasses_feature(crop: np.ndarray, min_confidence: float) -> DetailFeature:
    eye_band = _safe_region(crop, 0.10, 0.24, 0.18, 0.82)
    forehead = _safe_region(crop, 0.03, 0.10, 0.22, 0.78)
    edge = _edge_density(eye_band)
    darkness = _dark_ratio(eye_band, threshold=85)
    face_contrast = _color_contrast(eye_band, forehead)
    present = _clip01(edge * 0.48 + darkness * 0.32 + face_contrast * 0.20)
    absent = _clip01((1.0 - edge) * 0.45 + (1.0 - darkness) * 0.35 + (1.0 - face_contrast) * 0.20)
    return _binary_feature("glasses", present, absent, min_confidence)


def _cap_feature(crop: np.ndarray, min_confidence: float) -> DetailFeature:
    top_head = _safe_region(crop, 0.00, 0.13, 0.15, 0.85)
    face_band = _safe_region(crop, 0.14, 0.30, 0.18, 0.82)
    edge = _edge_density(top_head)
    darkness = _dark_ratio(top_head, threshold=95)
    contrast = _color_contrast(top_head, face_band)
    present = _clip01(contrast * 0.45 + darkness * 0.35 + edge * 0.20)
    absent = _clip01((1.0 - contrast) * 0.45 + (1.0 - darkness) * 0.35 + (1.0 - edge) * 0.20)
    return _binary_feature("cap", present, absent, min_confidence)


def _hood_feature(crop: np.ndarray, min_confidence: float) -> DetailFeature:
    left_head_side = _safe_region(crop, 0.06, 0.33, 0.02, 0.24)
    right_head_side = _safe_region(crop, 0.06, 0.33, 0.76, 0.98)
    face_center = _safe_region(crop, 0.12, 0.32, 0.32, 0.68)
    upper_torso = _safe_region(crop, 0.30, 0.48, 0.18, 0.82)
    side_signal = max(_color_contrast(left_head_side, face_center), _color_contrast(right_head_side, face_center))
    torso_signal = _color_contrast(upper_torso, face_center)
    edge_signal = max(_edge_density(left_head_side), _edge_density(right_head_side))
    present = _clip01(side_signal * 0.40 + torso_signal * 0.35 + edge_signal * 0.25)
    absent = _clip01((1.0 - side_signal) * 0.45 + (1.0 - torso_signal) * 0.35 + (1.0 - edge_signal) * 0.20)
    return _binary_feature("hood", present, absent, min_confidence)


def _long_hair_feature(crop: np.ndarray, min_confidence: float) -> DetailFeature:
    left_neck = _safe_region(crop, 0.20, 0.48, 0.05, 0.32)
    right_neck = _safe_region(crop, 0.20, 0.48, 0.68, 0.95)
    face_center = _safe_region(crop, 0.12, 0.30, 0.32, 0.68)
    shoulder = _safe_region(crop, 0.34, 0.52, 0.20, 0.80)
    side_dark = max(_dark_ratio(left_neck, threshold=95), _dark_ratio(right_neck, threshold=95))
    side_contrast = max(_color_contrast(left_neck, face_center), _color_contrast(right_neck, face_center))
    shoulder_contrast = max(_color_contrast(left_neck, shoulder), _color_contrast(right_neck, shoulder))
    present = _clip01(side_dark * 0.40 + side_contrast * 0.35 + shoulder_contrast * 0.25)
    absent = _clip01((1.0 - side_dark) * 0.35 + (1.0 - side_contrast) * 0.35 + (1.0 - shoulder_contrast) * 0.30)
    return _binary_feature("long_hair", present, absent, min_confidence)


def _beard_feature(crop: np.ndarray, min_confidence: float) -> DetailFeature:
    lower_face = _safe_region(crop, 0.22, 0.34, 0.30, 0.70)
    upper_face = _safe_region(crop, 0.13, 0.21, 0.30, 0.70)
    darkness = _dark_ratio(lower_face, threshold=105)
    texture = _laplacian_texture(lower_face)
    contrast = _color_contrast(lower_face, upper_face)
    present = _clip01(darkness * 0.42 + texture * 0.30 + contrast * 0.28)
    absent = _clip01((1.0 - darkness) * 0.45 + (1.0 - texture) * 0.25 + (1.0 - contrast) * 0.30)
    return _binary_feature("beard", present, absent, min_confidence)


def _watch_feature(crop: np.ndarray, min_confidence: float) -> DetailFeature:
    # Wrist zones are only a weak approximation because the current pipeline does
    # not run pose/keypoint estimation yet. The signal is intentionally low-weighted.
    left_wrist_zone = _safe_region(crop, 0.48, 0.86, 0.00, 0.22)
    right_wrist_zone = _safe_region(crop, 0.48, 0.86, 0.78, 1.00)
    center_body = _safe_region(crop, 0.42, 0.84, 0.30, 0.70)

    left_score = _clip01(_edge_density(left_wrist_zone) * 0.55 + _dark_ratio(left_wrist_zone, threshold=75) * 0.45)
    right_score = _clip01(_edge_density(right_wrist_zone) * 0.55 + _dark_ratio(right_wrist_zone, threshold=75) * 0.45)
    wrist_signal = max(left_score, right_score)
    body_contrast = max(_color_contrast(left_wrist_zone, center_body), _color_contrast(right_wrist_zone, center_body))
    present = _clip01(wrist_signal * 0.60 + body_contrast * 0.40)
    absent = _clip01((1.0 - wrist_signal) * 0.55 + (1.0 - body_contrast) * 0.45)
    return _binary_feature("watch", present, absent, min_confidence)


def _backpack_feature(crop: np.ndarray, min_confidence: float) -> DetailFeature:
    left_side = _safe_region(crop, 0.25, 0.75, 0.00, 0.20)
    right_side = _safe_region(crop, 0.25, 0.75, 0.80, 1.00)
    torso_center = _safe_region(crop, 0.28, 0.75, 0.25, 0.75)
    side_contrast = max(_color_contrast(left_side, torso_center), _color_contrast(right_side, torso_center))
    side_edges = max(_edge_density(left_side), _edge_density(right_side))
    side_dark = max(_dark_ratio(left_side, threshold=90), _dark_ratio(right_side, threshold=90))
    present = _clip01(side_contrast * 0.45 + side_edges * 0.30 + side_dark * 0.25)
    absent = _clip01((1.0 - side_contrast) * 0.45 + (1.0 - side_edges) * 0.25 + (1.0 - side_dark) * 0.30)
    return _binary_feature("backpack", present, absent, min_confidence)


def _feature_vector_from_features(features: dict[str, DetailFeature]) -> list[float]:
    vector: list[float] = []
    for spec in DETAIL_REGISTRY:
        feature = features.get(spec.name)
        if feature is None:
            vector.extend([0.0] * spec.length)
            continue
        values = feature.values[: spec.length]
        vector.extend([float(_clip01(value)) for value in values])
        if len(values) < spec.length:
            vector.extend([0.0] * (spec.length - len(values)))
    return vector


def extract_detail_snapshot(crop: np.ndarray | None, min_confidence: float = 0.55) -> DetailSnapshot | None:
    """Extract cheap, replaceable detail signals from a person crop.

    This is an MVP detail layer, not a biometric classifier. It extracts several
    weak cues from body regions: head accessories, hair/beard hints, upper/lower
    clothing colour, and texture signals that can represent shirt prints or
    patterns. The surrounding storage/matching contract is stable enough that a
    trained attribute classifier can replace these heuristics later.
    """
    if crop is None or crop.size == 0:
        return None
    height, width = crop.shape[:2]
    if width < 24 or height < 72:
        return None

    head = _safe_region(crop, 0.00, 0.34, 0.12, 0.88)
    upper = _safe_region(crop, 0.28, 0.66, 0.08, 0.92)
    lower = _safe_region(crop, 0.62, 1.00, 0.10, 0.90)

    features: dict[str, DetailFeature] = {
        "glasses": _glasses_feature(crop, min_confidence),
        "cap": _cap_feature(crop, min_confidence),
        "hood": _hood_feature(crop, min_confidence),
        "long_hair": _long_hair_feature(crop, min_confidence),
        "beard": _beard_feature(crop, min_confidence),
        "watch": _watch_feature(crop, min_confidence),
        "backpack": _backpack_feature(crop, min_confidence),
        "upper_color_hist": _vector_feature("upper_color_hist", _color_hist(upper), confidence=0.85 if upper is not None else 0.0),
        "lower_color_hist": _vector_feature("lower_color_hist", _color_hist(lower), confidence=0.80 if lower is not None else 0.0),
        "head_color_hist": _vector_feature("head_color_hist", _color_hist(head), confidence=0.65 if head is not None else 0.0),
        "shirt_print_texture": _vector_feature("shirt_print_texture", _shirt_print_texture(upper), confidence=0.65 if upper is not None else 0.0),
        "torso_region_profile": _vector_feature("torso_region_profile", _small_region_profile(upper), confidence=0.75 if upper is not None else 0.0),
    }

    vector = _feature_vector_from_features(features)
    reliable_values = [feature.confidence for feature in features.values() if feature.confidence > 0]
    reliability = _clip01(float(np.mean(reliable_values)) if reliable_values else 0.0)

    label_parts: list[str] = []
    for name in ("glasses", "cap", "hood", "long_hair", "beard", "backpack"):
        feature = features[name]
        if feature.status != "unknown":
            label_parts.append(f"{feature.name}:{feature.status}")
    label = ", ".join(label_parts[:4]) if label_parts else "details:regions"
    return DetailSnapshot(features=features, vector=vector, label=label, reliability=reliability)


def _vector_slices() -> dict[str, slice]:
    start = 0
    slices: dict[str, slice] = {}
    for spec in DETAIL_REGISTRY:
        end = start + spec.length
        slices[spec.name] = slice(start, end)
        start = end
    return slices


DETAIL_VECTOR_SLICES: dict[str, slice] = _vector_slices()
DETAIL_VECTOR_LENGTH: int = sum(spec.length for spec in DETAIL_REGISTRY)


def _cosine01(query: np.ndarray, stored: np.ndarray) -> float:
    q_norm = max(float(np.linalg.norm(query)), 1e-6)
    s_norm = max(float(np.linalg.norm(stored)), 1e-6)
    cosine = float(np.dot(query / q_norm, stored / s_norm))
    return _clip01((cosine + 1.0) / 2.0)


def detail_similarity_breakdown(
    query_vector: list[float] | np.ndarray | None,
    stored_vector: list[float] | np.ndarray | None,
) -> dict[str, Any]:
    if query_vector is None or stored_vector is None:
        return {"score": 0.5, "reason": "missing_detail_vector", "features": []}
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    stored = np.asarray(stored_vector, dtype=np.float32).reshape(-1)
    if query.size == 0 or stored.size == 0 or query.shape != stored.shape or query.size != DETAIL_VECTOR_LENGTH:
        return {"score": 0.5, "reason": "incompatible_detail_vector", "features": []}

    # Detail signals are intentionally conservative:
    # - unknown/weak details are neutral and do not influence the score;
    # - stable details can slightly support a visual match;
    # - volatile details (cap, hood, watch, print, backpack) are only weakly negative
    #   so appearance changes do not destroy a valid OSNet match.
    influence_sum = 0.0
    weight_sum = 0.0
    rows: list[dict[str, Any]] = []
    for spec in DETAIL_REGISTRY:
        slc = DETAIL_VECTOR_SLICES[spec.name]
        q_part = query[slc]
        s_part = stored[slc]
        ignored = False
        if spec.kind == "binary":
            q_strength = float(np.max(q_part))
            s_strength = float(np.max(s_part))
            if q_strength < 0.45 or s_strength < 0.45:
                feature_score = 0.5
                ignored = True
            else:
                feature_score = _cosine01(q_part, s_part)
        else:
            if float(np.linalg.norm(q_part)) < 1e-6 or float(np.linalg.norm(s_part)) < 1e-6:
                feature_score = 0.5
                ignored = True
            else:
                feature_score = _cosine01(q_part, s_part)

        effective_weight = float(spec.weight)
        if ignored:
            effective_weight = 0.0
        elif feature_score < 0.5 and spec.volatile:
            effective_weight *= 0.25
        elif feature_score < 0.5:
            effective_weight *= 0.50

        influence = (float(feature_score) - 0.5) * effective_weight
        influence_sum += influence
        weight_sum += effective_weight
        rows.append(
            {
                "name": spec.name,
                "label": spec.label,
                "group": spec.group,
                "kind": spec.kind,
                "weight": float(spec.weight),
                "effective_weight": float(effective_weight),
                "volatile": bool(spec.volatile),
                "score": float(feature_score),
                "influence": float(influence),
                "ignored": bool(ignored),
            }
        )
    if weight_sum <= 1e-6:
        return {"score": 0.5, "reason": "all_details_neutral_or_unknown", "features": rows}
    score = _clip01(0.5 + influence_sum / weight_sum)
    rows.sort(key=lambda row: abs(float(row["influence"])), reverse=True)
    return {"score": score, "reason": "ok_conservative_details", "features": rows}


def detail_similarity(query_vector: list[float] | np.ndarray | None, stored_vector: list[float] | np.ndarray | None) -> float:
    return float(detail_similarity_breakdown(query_vector, stored_vector).get("score", 0.5))


def combine_visual_and_detail_score(visual_score: float, detail_score: float | None, detail_weight: float) -> float:
    detail_weight = _clip01(detail_weight)
    visual_score = _clip01(visual_score)
    if detail_score is None:
        return visual_score
    # Treat 0.5 as neutral. Stable matching details can slightly boost a visual
    # match; contradictory details can slightly reduce it. This avoids a cap/no-cap
    # change destroying an otherwise good OSNet match.
    return _clip01(visual_score + (_clip01(detail_score) - 0.5) * detail_weight)


def compact_detail_influences(breakdown: dict[str, Any] | None, limit: int = 5) -> str:
    if not breakdown:
        return ""
    features = breakdown.get("features")
    if not isinstance(features, list):
        return ""
    parts: list[str] = []
    for item in features[:limit]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("name") or "feature")
        score = item.get("score")
        influence = item.get("influence")
        try:
            parts.append(f"{label}: {float(score):.2f} ({float(influence):+.3f})")
        except (TypeError, ValueError):
            continue
    return "; ".join(parts)
