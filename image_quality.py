"""
image_quality.py
Pre-flight image quality checks for the Vistone skin-tone pipeline.

Checks performed:
  - Minimum resolution
  - Overexposure (high L* median)
  - Underexposure (low L* median)
  - Blurriness via Laplacian variance
  - Face pose (yaw) estimation via MediaPipe FaceMesh landmark geometry

All functions accept an RGB numpy array (H×W×3, uint8).
"""

import cv2
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MIN_SHORT_SIDE   = 256     # pixels — below this, landmarks drift
BLUR_THRESHOLD   = 80.0    # Laplacian variance below this = blurry
L_OVEREXPOSED    = 94.0    # LAB L* median above this = washed-out
L_UNDEREXPOSED   = 14.0    # LAB L* median below this = too dark
MAX_YAW_DEGREES  = 28.0    # head turn beyond this = unreliable cheek mask


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _lab_median_L(rgb: np.ndarray) -> float:
    """Return median L* (0–100) of the image in CIE LAB."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(float)
    # OpenCV stores L in [0, 255] → rescale to [0, 100]
    return float(np.median(L)) * (100.0 / 255.0)


def _laplacian_variance(rgb: np.ndarray) -> float:
    """Measure image sharpness via Laplacian variance on luma."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _estimate_yaw(lm, img_shape: tuple) -> Optional[float]:
    """
    Estimate head yaw (horizontal rotation) from MediaPipe FaceMesh landmarks.

    Uses the horizontal distance between nose tip and facial midline
    (midpoint of left/right face edges at landmark 234 and 454) as a proxy.

    Returns yaw in degrees (positive = rightward turn) or None on failure.
    """
    try:
        h, w = img_shape[:2]

        # Landmark indices:
        #  4   = nose tip
        # 234  = left face edge (cheek)
        # 454  = right face edge (cheek)
        nose_x  = lm[4].x   * w
        left_x  = lm[234].x * w
        right_x = lm[454].x * w

        face_cx = (left_x + right_x) / 2.0
        face_w  = abs(right_x - left_x)

        if face_w < 1:
            return None

        # Normalised offset of nose from center
        offset_norm = (nose_x - face_cx) / (face_w / 2.0)
        # Map to degrees (empirical: offset_norm=1 ≈ 40°)
        yaw_deg = offset_norm * 40.0
        return float(yaw_deg)
    except Exception:
        return None


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def check_image_quality(
    rgb: np.ndarray,
    face_landmarks=None,
) -> dict:
    """
    Run all quality checks on an RGB image.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.
    face_landmarks : optional
        MediaPipe FaceMesh landmark list (if already computed, avoids re-run).

    Returns
    -------
    dict with keys:
        is_usable : bool   — False if any blocking issue is found
        warnings  : list[str]  — list of warning codes
        median_L  : float  — median L* (0–100)
        sharpness : float  — Laplacian variance
        yaw_deg   : float | None — estimated head yaw
    """
    h, w = rgb.shape[:2]
    warnings = []

    # 1. Resolution
    short_side = min(h, w)
    if short_side < MIN_SHORT_SIDE:
        warnings.append("LOW_RESOLUTION")

    # 2. Exposure (LAB L* channel)
    median_L = _lab_median_L(rgb)
    if median_L > L_OVEREXPOSED:
        warnings.append("OVEREXPOSED")
    elif median_L < L_UNDEREXPOSED:
        warnings.append("UNDEREXPOSED")

    # 3. Sharpness / blur
    sharpness = _laplacian_variance(rgb)
    if sharpness < BLUR_THRESHOLD:
        warnings.append("BLURRY")

    # 4. Face pose (only if landmarks provided)
    yaw_deg = None
    if face_landmarks is not None:
        yaw_deg = _estimate_yaw(face_landmarks, (h, w))
        if yaw_deg is not None and abs(yaw_deg) > MAX_YAW_DEGREES:
            warnings.append("NON_FRONTAL_FACE")

    # Blocking issues: any warning makes image unusable for high-confidence analysis.
    # We treat LOW_RESOLUTION, OVEREXPOSED, UNDEREXPOSED, BLURRY as hard blocks.
    # NON_FRONTAL_FACE is a soft warning (analysis still runs, confidence reduced).
    blocking = {"LOW_RESOLUTION", "OVEREXPOSED", "UNDEREXPOSED", "BLURRY"}
    is_usable = not bool(blocking.intersection(warnings))

    return {
        "is_usable": is_usable,
        "warnings":  warnings,
        "median_L":  round(median_L, 2),
        "sharpness": round(sharpness, 2),
        "yaw_deg":   round(yaw_deg, 1) if yaw_deg is not None else None,
    }


def quality_warning_message(quality_result: dict) -> str:
    """
    Return a user-friendly warning string from a quality result dict,
    or an empty string if no issues found.
    """
    msgs = {
        "LOW_RESOLUTION":   "Image resolution is too low. Use a photo of at least 256×256 px.",
        "OVEREXPOSED":      "Image appears overexposed. Try a photo in softer, natural light.",
        "UNDEREXPOSED":     "Image is too dark. Try better lighting or a brighter photo.",
        "BLURRY":           "Image is blurry. Use a sharp, in-focus photo.",
        "NON_FRONTAL_FACE": "Face appears turned to the side. Front-facing photos give better results.",
    }
    codes = quality_result.get("warnings", [])
    if not codes:
        return ""
    # Return the first blocking message, then soft warnings
    blocking = [c for c in codes if c != "NON_FRONTAL_FACE"]
    soft     = [c for c in codes if c == "NON_FRONTAL_FACE"]
    ordered  = blocking + soft
    return " ".join(msgs[c] for c in ordered if c in msgs)
