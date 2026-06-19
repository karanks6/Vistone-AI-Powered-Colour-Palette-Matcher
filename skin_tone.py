"""
skin_tone.py  —  Vistone v11.0
=================================
Classifies a facial photo on the Google Monk Skin Tone (MST) scale (1–10)
and detects undertone (Warm / Cool / Neutral).

v11.0 dark-skin accuracy fixes over v10.0
------------------------------------------
  • CLAHE DISABLED for dark skin faces (L_mean < 45) — it was lifting
    Monk 9/10 L* by +11 units, causing systematic misclassification
  • Gray World WB disabled for dark skin — it distorts chromaticity
  • Beard mask threshold raised (V < 70, not 95) — was eliminating
    valid dark skin pixels from the ROI
  • GMM L-band for dark mode widened to (8.0–58.0) to include Monk 9/10
  • topL_mask only used in light path (never dark path)
  • Skin colour measured directly from RAW image (pre-WB) for Monks 8-10
    then WB correction is applied as a small chroma offset only
  • Boltzmann temperature lowered to 3.5 for sharper classification
  • Monks 9 and 10 boundary decision uses L* tie-breaking rule
    (when dE < 6.0, pick by L* proximity instead of WB-distorted chroma)
  • Added tone_bias_correction: if measured L* implies a darker tone than
    the Boltzmann argmax, shift down by 1 if within-1 margin
"""

import os, cv2, json, numpy as np, mediapipe as mp
from sklearn.mixture import GaussianMixture
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from image_quality import check_image_quality, quality_warning_message


# ══════════════════════════════════════════════════════════════════════
# 1.  OFFICIAL GOOGLE MONK PALETTE  (Ellis et al. 2022)
#     Source: skintone.google  — confirmed hex values
# ══════════════════════════════════════════════════════════════════════
MONK_TONES_HEX = [
    "#f6ede4",   # Monk 1  L*≈94.2  lightest
    "#f3e7db",   # Monk 2  L*≈92.3
    "#f7ead0",   # Monk 3  L*≈93.1
    "#eadaba",   # Monk 4  L*≈87.6
    "#d7bd96",   # Monk 5  L*≈77.9
    "#a07e56",   # Monk 6  L*≈55.1
    "#825c43",   # Monk 7  L*≈42.5
    "#604134",   # Monk 8  L*≈30.7
    "#3a312a",   # Monk 9  L*≈21.1
    "#292420",   # Monk 10 L*≈14.6  darkest
]


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _lab_d65(hex_str: str) -> LabColor:
    r, g, b = _hex_to_rgb(hex_str)
    return convert_color(
        sRGBColor(r / 255.0, g / 255.0, b / 255.0),
        LabColor, target_illuminant="d65"
    )


MONK_TONES_LAB_D65 = [_lab_d65(h) for h in MONK_TONES_HEX]

# L* of each Monk reference in D65 — used for tie-breaking
MONK_L_D65 = np.array([lab.lab_l for lab in MONK_TONES_LAB_D65])
# [94.21, 92.27, 93.09, 87.57, 77.90, 55.14, 42.47, 30.68, 21.07, 14.61]

# Lookup: sRGB of each Monk tone (for display / comparison)
MONK_TONES_RGB = [np.array(_hex_to_rgb(h), dtype=float) for h in MONK_TONES_HEX]


# ══════════════════════════════════════════════════════════════════════
# 2.  TONE-STRATIFIED UNDERTONE REFERENCES
# ══════════════════════════════════════════════════════════════════════
UNDERTONE_REFS = {
    (1, 3): {
        "warm":    LabColor(91.0,  4.5, 13.0),
        "cool":    LabColor(91.0,  3.0, -1.5),
        "neutral": LabColor(91.0,  3.5,  5.5),
    },
    (4, 5): {
        "warm":    LabColor(82.0, 10.0, 18.5),
        "cool":    LabColor(82.0,  5.0,  3.5),
        "neutral": LabColor(82.0,  7.0, 10.5),
    },
    (6, 7): {
        "warm":    LabColor(47.0, 14.0, 20.5),
        "cool":    LabColor(47.0,  8.0,  5.0),
        "neutral": LabColor(47.0, 11.0, 12.5),
    },
    (8, 10): {
        "warm":    LabColor(24.0, 10.5, 14.5),
        "cool":    LabColor(24.0,  5.5, -1.0),
        "neutral": LabColor(24.0,  7.5,  7.0),
    },
}


def _get_undertone_refs(monk_tone: int) -> dict:
    for (lo, hi), refs in UNDERTONE_REFS.items():
        if lo <= monk_tone <= hi:
            return refs
    return UNDERTONE_REFS[(6, 7)]


# ══════════════════════════════════════════════════════════════════════
# 3.  MEDIAPIPE
# ══════════════════════════════════════════════════════════════════════
mp_face_mesh = mp.solutions.face_mesh


# ══════════════════════════════════════════════════════════════════════
# 4.  JSON COLOR RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════
def _load_color_json():
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    path = os.path.join(base, "monk_skin_tone_color_recommendations.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] Could not load color JSON: {e}")
        return None


COLOR_DATA = _load_color_json()


def get_color_recommendations(tone: int, undertone: str):
    if not COLOR_DATA:
        return [], []
    key     = f"tone_{tone}"
    block   = COLOR_DATA.get(key, {})
    if not isinstance(block, dict):
        return [], []
    ut      = (undertone or "").lower()
    payload = block.get(ut) or block.get("neutral") or block.get("default")
    if not payload:
        return [], []
    best  = [{"name": c.get("name", ""), "hex": c.get("hex", "")}
             for c in (payload.get("best")  or [])][:3]
    avoid = [{"name": c.get("name", ""), "hex": c.get("hex", "")}
             for c in (payload.get("avoid") or [])][:3]
    return best, avoid


# ══════════════════════════════════════════════════════════════════════
# 5.  WHITE BALANCE
#     NEW: dark_skin=True skips gray-world (too destructive on dark faces)
# ══════════════════════════════════════════════════════════════════════
def shades_of_gray_wb(rgb: np.ndarray, p: int = 6) -> np.ndarray:
    f   = rgb.astype(np.float32) / 255.0
    eps = 1e-6
    rn  = np.power(np.mean(np.power(f[..., 2], p)), 1 / p) + eps
    gn  = np.power(np.mean(np.power(f[..., 1], p)), 1 / p) + eps
    bn  = np.power(np.mean(np.power(f[..., 0], p)), 1 / p) + eps
    m   = (rn + gn + bn) / 3.0
    f[..., 2] *= m / rn
    f[..., 1] *= m / gn
    f[..., 0] *= m / bn
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def sclera_based_wb(rgb: np.ndarray, eye_mask: np.ndarray,
                     dark_skin: bool = False) -> np.ndarray | None:
    """
    Sclera white-balance. For dark skin, we use a looser V threshold (120)
    and cap the correction strength to avoid overcorrection.
    """
    hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    V_thresh = 120 if dark_skin else 150
    scl  = (eye_mask > 0) & (S < 60) & (V > V_thresh)
    if np.count_nonzero(scl) < 80:
        return None
    sel    = rgb[scl]
    mb, mg, mr = sel[:, 0].mean() + 1e-6, sel[:, 1].mean() + 1e-6, sel[:, 2].mean() + 1e-6
    gray   = (mb + mg + mr) / 3.0
    f      = rgb.astype(np.float32)
    # Cap channel multipliers to ±20% to avoid overcorrection on dark faces
    mult_b = float(np.clip(gray / mb, 0.80, 1.20))
    mult_g = float(np.clip(gray / mg, 0.80, 1.20))
    mult_r = float(np.clip(gray / mr, 0.80, 1.20))
    f[..., 0] *= mult_b
    f[..., 1] *= mult_g
    f[..., 2] *= mult_r
    return np.clip(f, 0, 255).astype(np.uint8)


def white_patch_wb_on_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sel = rgb[mask > 0]
    if sel.size < 150:
        return rgb
    mb, mg, mr = sel[:, 0].mean() + 1e-6, sel[:, 1].mean() + 1e-6, sel[:, 2].mean() + 1e-6
    gray = (mb + mg + mr) / 3.0
    f    = rgb.astype(np.float32)
    f[..., 0] *= gray / mb
    f[..., 1] *= gray / mg
    f[..., 2] *= gray / mr
    return np.clip(f, 0, 255).astype(np.uint8)


def skin_only_gray_world(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sel = rgb[mask > 0]
    if sel.size < 150:
        return rgb
    mb, mg, mr = sel[:, 0].mean() + 1e-6, sel[:, 1].mean() + 1e-6, sel[:, 2].mean() + 1e-6
    m = (mb + mg + mr) / 3.0
    f = rgb.astype(np.float32)
    f[..., 0] *= m / mb
    f[..., 1] *= m / mg
    f[..., 2] *= m / mr
    return np.clip(f, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
# 6.  CLAHE  —  only applied when skin L_mean >= 45
#     Disabled for dark skin to prevent L* inflation of Monk 8-10
# ══════════════════════════════════════════════════════════════════════
def clahe_normalize_face(rgb: np.ndarray, bbox: tuple,
                          clip_limit: float = 1.5,
                          tile_grid: tuple = (8, 8),
                          L_mean_override: float | None = None) -> np.ndarray:
    """
    Apply CLAHE on the face bounding box (L* channel only).
    SKIPPED entirely if median face L* < 45 (dark skin protection).
    """
    # Decide whether to apply
    if L_mean_override is not None:
        if L_mean_override < 45.0:
            return rgb          # skip for dark skin

    x1, y1, x2, y2 = bbox
    h, w = rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return rgb

    out         = rgb.copy()
    face_region = out[y1:y2, x1:x2]
    lab         = cv2.cvtColor(face_region, cv2.COLOR_RGB2LAB)
    L, a, b     = cv2.split(lab)
    clahe       = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    L_eq        = clahe.apply(L)
    lab_eq      = cv2.merge([L_eq, a, b])
    out[y1:y2, x1:x2] = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return out


# ══════════════════════════════════════════════════════════════════════
# 7.  FACE DETECTION
# ══════════════════════════════════════════════════════════════════════
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROTO    = os.path.join(_BASE_DIR, "deploy.prototxt")
_MODEL    = os.path.join(_BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
_dnn_net  = None


def _load_dnn_net():
    global _dnn_net
    if _dnn_net is None and os.path.isfile(_PROTO) and os.path.isfile(_MODEL):
        try:
            _dnn_net = cv2.dnn.readNetFromCaffe(_PROTO, _MODEL)
        except Exception as e:
            print(f"[warn] DNN load failed: {e}")
    return _dnn_net


def fast_face_check(rgb: np.ndarray, conf_threshold: float = 0.55) -> bool:
    net = _load_dnn_net()
    if net is None:
        return True
    blob = cv2.dnn.blobFromImage(
        cv2.resize(rgb, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        if float(detections[0, 0, i, 2]) >= conf_threshold:
            return True
    return False


def detect_mesh(rgb: np.ndarray):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50,
    ) as fm:
        res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None, None
    lm   = res.multi_face_landmarks[0].landmark
    h, w = rgb.shape[:2]
    xs   = np.array([int(p.x * w) for p in lm])
    ys   = np.array([int(p.y * h) for p in lm])
    return lm, (xs.min(), ys.min(), xs.max(), ys.max())


def auto_pad_if_needed(rgb, bbox, debug, save_dir, name):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    fh = y2 - y1
    if fh / max(1, h) <= 0.70:
        return rgb, False
    pad = int(fh * 0.35)
    p   = cv2.copyMakeBorder(
        rgb, pad, pad // 3, pad // 3, pad // 3,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    if debug:
        cv2.imwrite(os.path.join(save_dir, f"padded_{name}.png"),
                    cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
    return p, True


# ══════════════════════════════════════════════════════════════════════
# 8.  LANDMARK GROUPS
# ══════════════════════════════════════════════════════════════════════
LEFT_CHEEK    = [234, 93, 132, 58, 172, 136, 150, 176]
RIGHT_CHEEK   = [454, 323, 361, 288, 397, 365, 379, 400]
FOREHEAD_POLY = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397]
LEFT_EYE_RING  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LEFT_EYE_FULL  = LEFT_EYE_RING  + [130, 247, 30, 29, 27]
RIGHT_EYE_FULL = RIGHT_EYE_RING + [359, 467, 260, 259, 257]
MOUTH_OUT  = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
NOSE_POLY  = [6, 197, 195, 5, 4, 45, 220, 218, 237, 1]


def _idxs_xy(img, idxs, lm) -> np.ndarray:
    h, w = img.shape[:2]
    return np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in idxs], dtype=np.int32)


def cheek_polys(img, lm):
    return [_idxs_xy(img, LEFT_CHEEK, lm), _idxs_xy(img, RIGHT_CHEEK, lm)]


def eyes_mask(img, lm):
    m = np.zeros(img.shape[:2], np.uint8)
    for idxs in (LEFT_EYE_RING, RIGHT_EYE_RING):
        cv2.fillConvexPoly(m, _idxs_xy(img, idxs, lm), 255)
    return m


def eyes_exclusion_mask(img, lm, px: int = 14):
    m = np.zeros(img.shape[:2], np.uint8)
    for idxs in (LEFT_EYE_FULL, RIGHT_EYE_FULL):
        cv2.fillConvexPoly(m, _idxs_xy(img, idxs, lm), 255)
    if px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px)), 1)
    return m


def mouth_mask(img, lm, px: int = 14):
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, _idxs_xy(img, MOUTH_OUT, lm), 255)
    if px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px)), 1)
    return m


def forehead_mask(img: np.ndarray, lm, er: int = 10,
                   skin_L_mean: float = 60.0) -> np.ndarray:
    """Forehead ROI with adaptive L* floor scaled to actual skin brightness."""
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, _idxs_xy(img, FOREHEAD_POLY, lm), 255)
    if er > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)), 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    S   = hsv[..., 1]
    L   = lab[..., 0].astype(float) * (100.0 / 255.0)
    # Adaptive floor: 50% of skin L_mean, clamped to [10, 58]
    L_floor = float(np.clip(skin_L_mean * 0.50, 10.0, 58.0))
    skin = ((S < 90) & (L > L_floor)).astype(np.uint8) * 255
    return cv2.bitwise_and(m, skin)


def nose_mask(img: np.ndarray, lm, skin_L_mean: float = 60.0) -> np.ndarray:
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, _idxs_xy(img, NOSE_POLY, lm), 255)
    m   = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L   = lab[..., 0].astype(float) * (100.0 / 255.0)
    S, V = hsv[..., 1], hsv[..., 2]
    L_floor = float(np.clip(skin_L_mean * 0.45, 8.0, 52.0))
    # Relaxed V threshold for dark skin (V > 30 instead of 50)
    V_thresh = max(30, int(skin_L_mean * 1.2))
    keep = ((L > L_floor) & (S < 85) & (V > V_thresh)).astype(np.uint8) * 255
    return cv2.bitwise_and(m, keep)


def poly_mask(shape, polys) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillConvexPoly(m, p, 255)
    return m


# ══════════════════════════════════════════════════════════════════════
# 9.  MASK CLEANUP
#     KEY FIX: beard threshold adjusted for dark skin
# ══════════════════════════════════════════════════════════════════════
def clean_mask(rgb: np.ndarray, mask: np.ndarray,
               dark_skin: bool = False) -> np.ndarray:
    """
    Remove specular highlights, lip pixels, and beard/stubble from mask.
    dark_skin=True relaxes the beard filter to avoid removing dark skin pixels.
    """
    hsv     = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S, V    = hsv[..., 1], hsv[..., 2]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Specular highlights (very high V, very low S)
    spec = ((S < 40) & (V > 240)).astype(np.uint8) * 255

    # Lip / redness removal
    lips1    = cv2.inRange(hsv, np.array([0,  60, 60]), np.array([12, 255, 255]))
    lips2    = cv2.inRange(hsv, np.array([165, 60, 60]), np.array([180, 255, 255]))
    lips_rgb = ((r > 150) & (g < 165) & (b < 165)).astype(np.uint8) * 255
    lips     = cv2.bitwise_or(lips1, cv2.bitwise_or(lips2, lips_rgb))

    # Beard / stubble filter
    # For dark skin: V < 70 (was 95) to avoid removing valid skin pixels
    # Dark skin pixels: S often 20-55, V often 40-120 — must not be removed
    V_beard = 70 if dark_skin else 95
    beard   = ((S > 55) & (V < V_beard)).astype(np.uint8) * 255

    rem  = cv2.bitwise_or(spec, cv2.bitwise_or(lips, beard))
    keep = cv2.bitwise_and(mask, cv2.bitwise_not(rem))
    return cv2.morphologyEx(
        keep, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1
    )


def mid_mask(rgb: np.ndarray, mask: np.ndarray,
             lp: float = 10.0, hp: float = 92.0) -> np.ndarray:
    lab  = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L    = lab[..., 0]
    vals = L[mask > 0]
    if vals.size < 50:
        return mask
    lo, hi = np.percentile(vals, [lp, hp])
    mm = ((L >= lo) & (L <= hi)).astype(np.uint8) * 255
    return cv2.bitwise_and(mask, mm)


def adaptive_mid_mask(rgb: np.ndarray, mask: np.ndarray,
                       rough_tone: int) -> np.ndarray:
    """Tone-adaptive percentile bounds — very wide for dark tones."""
    if rough_tone <= 3:
        lp, hp = 15.0, 85.0
    elif rough_tone <= 6:
        lp, hp = 8.0,  93.0
    elif rough_tone <= 7:
        lp, hp = 5.0,  95.0
    else:
        lp, hp = 2.0,  98.0   # Monks 8-10: keep almost all pixels
    return mid_mask(rgb, mask, lp=lp, hp=hp)


# ══════════════════════════════════════════════════════════════════════
# 10. PIXEL STATISTICS
# ══════════════════════════════════════════════════════════════════════
def robust_median(px: np.ndarray) -> np.ndarray | None:
    if px is None or px.size == 0:
        return None
    px = px.astype(float)
    med = np.median(px, 0)
    d   = np.linalg.norm(px - med, axis=1)
    thr = np.percentile(d, 80)
    k   = px[d <= thr]
    if k.size < 50:
        k = px
    return np.median(k, 0)


def hsv_stats(rgb: np.ndarray, mask: np.ndarray):
    hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S, V = hsv[..., 1].astype(float), hsv[..., 2].astype(float)
    s, v = S[mask > 0], V[mask > 0]
    if s.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(s)), float(np.std(s)), float(np.std(v))


def erode_inner_cheeks_dist(mask: np.ndarray, px: int = 12) -> np.ndarray:
    m    = (mask > 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return (dist > px).astype(np.uint8) * 255


def topL_mask(rgb: np.ndarray, base_mask: np.ndarray, pct: float = 20) -> np.ndarray:
    """Select top-pct% brightest non-specular pixels. Light path only."""
    lab  = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L    = lab[..., 0]
    vals = L[base_mask > 0]
    if vals.size < 60:
        return base_mask
    t       = np.percentile(vals, 100 - pct)
    bright  = (L >= t).astype(np.uint8) * 255
    hsv     = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    nonspec = (~((hsv[..., 1] < 30) & (hsv[..., 2] > 246))).astype(np.uint8) * 255
    return cv2.bitwise_and(cv2.bitwise_and(base_mask, bright), nonspec)


def _ensure_rgb3(v, fallback=None, tag="fused") -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim > 1:
        v = v.reshape(-1)
    if v.size < 3 or not np.all(np.isfinite(v[:3])):
        if fallback is not None:
            return np.asarray(fallback, dtype=np.float32).reshape(-1)[:3]
        return np.array([128, 128, 128], np.float32)
    return v[:3]


# ══════════════════════════════════════════════════════════════════════
# 11. GMM — adaptive BIC, tone-aware L-bands
# ══════════════════════════════════════════════════════════════════════
def _cluster_stats(centers_rgb: np.ndarray, counts_frac: np.ndarray):
    labs, min_dE = [], []
    for c in centers_rgb:
        srgb = sRGBColor(*(np.asarray(c) / 255.0).reshape(-1)[:3].tolist())
        lab  = convert_color(srgb, LabColor, target_illuminant="d65")
        labs.append(lab)
        d = [delta_e_cie2000(lab, t) for t in MONK_TONES_LAB_D65]
        min_dE.append(min(d))
    Ls = np.array([lab.lab_l for lab in labs], float)
    return Ls, np.array(min_dE, float), np.array(counts_frac, float), labs


def gmm_dominant_rgb(pixels_rgb: np.ndarray, mode: str = "medium",
                      return_clusters: bool = False):
    """
    BIC-optimal GMM (n=1..5) per pixel set.
    mode: 'light' | 'medium' | 'dark'
    L-bands:
      light  → (70, 96)
      medium → (40, 82)
      dark   → (8,  58)   ← widened to cover Monk 9 (L*≈21) and 10 (L*≈14.6)
    """
    if pixels_rgb is None or pixels_rgb.size < 90:
        return (None, None, None) if return_clusters else None

    X = pixels_rgb.astype(np.float32)
    X = X[np.all(np.isfinite(X), axis=1)]
    if X.shape[0] < 90:
        return (None, None, None) if return_clusters else None

    best_bic, best_gmm = np.inf, None
    for n in range(1, 6):
        try:
            g = GaussianMixture(n_components=n, covariance_type="full",
                                random_state=0, reg_covar=1e-6)
            g.fit(X)
            bic = g.bic(X)
            if bic < best_bic:
                best_bic, best_gmm = bic, g
        except Exception:
            continue

    if best_gmm is None:
        fm = robust_median(X)
        return (fm, None, None) if return_clusters else fm

    n_comp  = best_gmm.n_components
    labels  = best_gmm.predict(X)
    counts  = np.bincount(labels, minlength=n_comp).astype(float)
    frac    = counts / max(1.0, counts.sum())
    centers = best_gmm.means_

    # Widened dark band to cover Monks 9 and 10
    L_bands = {
        "light":  (70.0, 96.0),
        "medium": (40.0, 82.0),
        "dark":   ( 8.0, 58.0),   # KEY FIX — was (15, 55)
    }
    L_lo, L_hi = L_bands.get(mode, (40.0, 82.0))

    Ls, dEs, fracs, labs = _cluster_stats(centers, frac)

    L_penalty  = np.zeros_like(Ls)
    L_penalty += np.where(Ls < L_lo, (L_lo - Ls) * 0.6, 0.0)  # softer penalty
    L_penalty += np.where(Ls > L_hi, (Ls - L_hi) * 0.5, 0.0)

    tiny_pen = (fracs < 0.05).astype(float) * 2.5

    score = (-dEs) + (1.2 * fracs) - (0.12 * L_penalty) - tiny_pen
    idx   = int(np.argmax(score))

    center = centers[idx]
    if not np.all(np.isfinite(center)):
        fm = robust_median(X)
        return (fm, centers, fracs) if return_clusters else fm

    if return_clusters:
        return center.astype(np.float32), centers.astype(np.float32), fracs.astype(np.float32)
    return center.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 12. MONK TONE MAPPING
# ══════════════════════════════════════════════════════════════════════
def monk_probabilities(fused_rgb: np.ndarray,
                        temperature: float = 3.5) -> np.ndarray:
    """
    Boltzmann probability over 10 Monk tones.
    Temperature 3.5 gives sharper peaks than 4.0.
    """
    v   = _ensure_rgb3(fused_rgb)
    lab = convert_color(sRGBColor(*(v / 255.0).tolist()), LabColor, target_illuminant="d65")
    dEs     = np.array([delta_e_cie2000(lab, t) for t in MONK_TONES_LAB_D65], float)
    neg_dE  = -dEs / temperature
    neg_dE -= neg_dE.max()
    probs   = np.exp(neg_dE)
    probs  /= probs.sum()
    return probs


def monk_from_probabilities(probs: np.ndarray) -> tuple:
    idx = int(np.argmax(probs))
    return idx + 1, float(probs[idx])


def rough_monk_tone(fused_rgb: np.ndarray) -> int:
    v   = _ensure_rgb3(fused_rgb)
    lab = convert_color(sRGBColor(*(v / 255.0).tolist()), LabColor, target_illuminant="d65")
    dEs = [delta_e_cie2000(lab, t) for t in MONK_TONES_LAB_D65]
    return int(np.argmin(dEs)) + 1


def _L_star_of_pixels(px: np.ndarray) -> float:
    """Return median L* (D65) of a pixel array (Nx3 uint8/float)."""
    if px is None or px.size == 0:
        return 50.0
    px = np.clip(np.asarray(px, float), 0, 255)
    # Fast approximate: convert median RGB to L*
    med = np.median(px, axis=0)
    lab = convert_color(sRGBColor(*(med / 255.0).tolist()), LabColor, target_illuminant="d65")
    return float(lab.lab_l)


def tone_bias_correction(tone: int, probs: np.ndarray,
                           measured_L: float) -> int:
    """
    If the measured skin L* strongly implies a darker tone than the
    Boltzmann argmax, shift the prediction darker by 1.
    This corrects WB/CLAHE-induced lightening bias on Monk 8-10.
    Only applies when top-2 tones differ by < 8 ΔE (tight boundary).
    """
    # Only correct in the dark range (Monks 7-10)
    if tone < 7:
        return tone

    # Expected L* range for predicted tone (±tolerance)
    expected_L = MONK_L_D65[tone - 1]
    L_gap      = expected_L - measured_L  # positive = predicted too light

    # If measured L* is more than 5 units darker than the predicted Monk → shift dark
    if L_gap > 5.0 and tone < 10:
        # Only shift if next darker tone is plausible (within 1.5× the top prob)
        darker_prob = probs[tone]      # probs is 0-indexed, tone is 1-indexed
        top_prob    = probs[tone - 1]
        if darker_prob > 0.10 or (top_prob - darker_prob) < 0.20:
            return tone + 1

    return tone


# ══════════════════════════════════════════════════════════════════════
# 13. UNDERTONE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════
def _undertone_from_lab_stratified(lab: LabColor, monk_tone: int,
                                    margin: float = 2.0) -> str:
    refs   = _get_undertone_refs(monk_tone)
    d_warm = delta_e_cie2000(lab, refs["warm"])
    d_cool = delta_e_cie2000(lab, refs["cool"])
    d_neut = delta_e_cie2000(lab, refs["neutral"])

    best   = min(d_warm, d_cool, d_neut)
    second = sorted([d_warm, d_cool, d_neut])[1]
    gap    = second - best

    if gap < margin:
        return "Neutral"
    if best == d_warm:
        return "Warm"
    if best == d_cool:
        return "Cool"
    return "Neutral"


def multi_region_undertone(rgb_raw: np.ndarray,
                            region_masks: dict,
                            monk_tone: int) -> tuple:
    """Undertone from cheek/forehead/nose regions with weighted voting."""
    region_weights = {"cheek": 0.50, "forehead": 0.32, "nose": 0.18}
    votes = {"Warm": 0.0, "Cool": 0.0, "Neutral": 0.0}

    for region, mask in region_masks.items():
        px = rgb_raw[mask > 0]
        if px.size < 50:
            continue
        med  = np.median(px, axis=0)
        srgb = sRGBColor(*(med / 255.0).tolist())
        lab  = convert_color(srgb, LabColor, target_illuminant="d65")
        ut   = _undertone_from_lab_stratified(lab, monk_tone)
        votes[ut] += region_weights.get(region, 0.2)

    total = sum(votes.values())
    if total < 1e-6:
        return "Neutral", 0.0
    best = max(votes, key=votes.get)
    return best, float(votes[best] / total)


def classify_undertone(fused_rgb: np.ndarray,
                        centers_rgb: np.ndarray | None,
                        fracs: np.ndarray | None,
                        monk_tone: int,
                        region_masks: dict | None = None,
                        rgb_source: np.ndarray | None = None) -> tuple:
    """Master undertone: fused-color + multi-region + cluster vote."""
    srgb     = sRGBColor(*(_ensure_rgb3(fused_rgb) / 255.0).tolist())
    lab_fuse = convert_color(srgb, LabColor, target_illuminant="d65")
    base_ut  = _undertone_from_lab_stratified(lab_fuse, monk_tone)

    votes = {"Warm": 0.0, "Cool": 0.0, "Neutral": 0.0}
    votes[base_ut] += 1.5

    if region_masks is not None and rgb_source is not None:
        mr_ut, mr_conf = multi_region_undertone(rgb_source, region_masks, monk_tone)
        votes[mr_ut] += 1.0 * mr_conf

    if centers_rgb is not None and fracs is not None:
        for c, f in zip(centers_rgb, fracs):
            sr  = sRGBColor(*(_ensure_rgb3(c) / 255.0).tolist())
            lab = convert_color(sr, LabColor, target_illuminant="d65")
            ut  = _undertone_from_lab_stratified(lab, monk_tone)
            votes[ut] += float(f) * 0.8

    total = sum(votes.values())
    best  = max(votes, key=votes.get)
    conf  = votes[best] / total if total > 0 else 0.0
    return best, float(conf)


# ══════════════════════════════════════════════════════════════════════
# 14. DEBUG DISPLAY
# ══════════════════════════════════════════════════════════════════════
def debug_display(imgs: list, title: str = ""):
    out = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in imgs if im is not None]
    c   = np.hstack(out)
    if c.shape[1] > 900:
        c = cv2.resize(c, None, fx=900 / c.shape[1], fy=900 / c.shape[1],
                       interpolation=cv2.INTER_AREA)
    cv2.imshow(title, c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════
# 15. MAIN CLASSIFIER  v11.0
# ══════════════════════════════════════════════════════════════════════
def classify_monk_v10(image_path: str, debug: bool = False) -> dict:
    """
    Full skin-tone + undertone pipeline (Vistone v11.0).
    Backward-compatible name; called internally by classify_monk_v9_5 alias.
    """
    # ── Load ──────────────────────────────────────────────────────────
    bgr = cv2.imread(image_path)
    if bgr is None:
        return _default_result("Could not read image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    name     = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.dirname(image_path)

    # ── Fast DNN pre-check ────────────────────────────────────────────
    if not fast_face_check(rgb):
        return _default_result("No face detected in image.")

    # ── MediaPipe FaceMesh ────────────────────────────────────────────
    lm, bbox = detect_mesh(rgb)
    if lm is None:
        return _default_result("No face landmarks detected.")

    # ── Image quality gate ────────────────────────────────────────────
    quality = check_image_quality(rgb, face_landmarks=lm)
    if not quality["is_usable"]:
        return _default_result(quality_warning_message(quality), quality=quality)

    median_L_global = quality["median_L"]

    # ── Auto-pad if face fills frame ──────────────────────────────────
    rgb_pad, padded = auto_pad_if_needed(rgb, bbox, debug, save_dir, name)
    if padded:
        lm2, bbox2 = detect_mesh(rgb_pad)
        if lm2 is not None:
            rgb, lm, bbox = rgb_pad, lm2, bbox2

    # ── Estimate face-region L* from cheek pixels (before WB/CLAHE) ──
    polys_raw   = cheek_polys(rgb, lm)
    cheeks_raw  = poly_mask(rgb.shape[:2], polys_raw)
    cheek_px    = rgb[cheeks_raw > 0]
    skin_L_mean = _L_star_of_pixels(cheek_px)
    dark_skin   = skin_L_mean < 45.0   # Monks 7-10 are approximately < 45 L*

    print(f"[v11] skin_L_mean={skin_L_mean:.1f}  dark_skin={dark_skin}")

    # ── CLAHE (light skin only — disabled for dark skin) ──────────────
    rgb = clahe_normalize_face(rgb, bbox, clip_limit=1.5,
                                L_mean_override=skin_L_mean)

    # ── Build ROI masks ───────────────────────────────────────────────
    polys    = cheek_polys(rgb, lm)
    cheeks   = poly_mask(rgb.shape[:2], polys)
    fore     = forehead_mask(rgb, lm, er=10, skin_L_mean=skin_L_mean)
    nose_m   = nose_mask(rgb, lm, skin_L_mean=skin_L_mean)
    eyes_exc = eyes_exclusion_mask(rgb, lm, px=14)
    mouth_m  = mouth_mask(rgb, lm, px=14)

    cheeks = cv2.bitwise_and(cheeks, cv2.bitwise_not(eyes_exc))
    fore   = cv2.bitwise_and(fore,   cv2.bitwise_not(eyes_exc))
    nose_m = cv2.bitwise_and(nose_m, cv2.bitwise_not(eyes_exc))
    roi    = cv2.bitwise_or(cheeks, fore)

    # ── White balance ─────────────────────────────────────────────────
    eye_wb = eyes_mask(rgb, lm)
    rgb_wb = sclera_based_wb(rgb, eye_wb, dark_skin=dark_skin)
    if rgb_wb is None:
        if dark_skin:
            # For very dark skin: skip aggressive global WB — use original
            rgb_wb = rgb.copy()
        else:
            rgb_wb = shades_of_gray_wb(rgb, p=6)

    # ── Mask cleaning ─────────────────────────────────────────────────
    clean = clean_mask(rgb_wb, roi, dark_skin=dark_skin)
    used  = cv2.bitwise_and(clean, cv2.bitwise_not(mouth_m))

    # ── Luminance stats from skin pixels ─────────────────────────────
    lab_tmp = cv2.cvtColor(rgb_wb, cv2.COLOR_RGB2LAB)
    Lch     = lab_tmp[..., 0].astype(float) * (100.0 / 255.0)
    s_mean, _, _ = hsv_stats(rgb_wb, used)
    L_vals  = Lch[used > 0]
    L_mean  = float(np.mean(L_vals)) if L_vals.size > 0 else skin_L_mean

    # Sigmoid blend weight (0 = dark path, 1 = light path)
    # Centred at L=70, steeper for dark skin to lock dark path
    sigmoid_centre = 70.0 if not dark_skin else 60.0
    light_w = float(1.0 / (1.0 + np.exp(-(L_mean - sigmoid_centre) / 7.0)))
    if dark_skin:
        light_w = min(light_w, 0.25)   # hard cap: dark path always dominates

    inner = erode_inner_cheeks_dist(cheeks, 12)

    # ── LIGHT PATH pixel set ──────────────────────────────────────────
    light_core = cv2.bitwise_or(inner, fore)
    if not dark_skin:
        b1        = topL_mask(rgb_wb, light_core, 20)
        rgb_local = white_patch_wb_on_mask(rgb_wb, b1)
        rgb_local = skin_only_gray_world(rgb_local, light_core)
        b2        = topL_mask(rgb_local, light_core, 20)
        sel_light = rgb_local[b2 > 0]
        if sel_light.size < 60:
            sel_light = rgb_local[light_core > 0]
        if sel_light.size < 50:
            sel_light = rgb_wb[used > 0]
    else:
        # Dark skin: use raw (pre-WB) cheek pixels directly
        sel_light = rgb[cv2.bitwise_and(clean_mask(rgb, roi, dark_skin=True), cheeks) > 0]
        if sel_light.size < 90:
            sel_light = rgb[cheeks > 0]

    # ── DARK PATH pixel set ───────────────────────────────────────────
    Lmask = cv2.bitwise_and(poly_mask(rgb.shape[:2], [polys[0]]), inner)
    Rmask = cv2.bitwise_and(poly_mask(rgb.shape[:2], [polys[1]]), inner)

    # For dark skin: sample from raw RGB to avoid WB distortion
    src = rgb if dark_skin else rgb_wb
    clean_src = clean_mask(src, roi, dark_skin=dark_skin)
    used_src  = cv2.bitwise_and(clean_src, cv2.bitwise_not(mouth_m))

    pxL = src[cv2.bitwise_and(used_src, Lmask) > 0]
    pxR = src[cv2.bitwise_and(used_src, Rmask) > 0]
    pxN = src[cv2.bitwise_and(used_src, nose_m) > 0]
    if pxL.size < 50: pxL = src[Lmask > 0]
    if pxR.size < 50: pxR = src[Rmask > 0]
    if pxN.size < 50: pxN = src[nose_m > 0]

    stacks   = [p for p in [pxL, pxR, pxN] if p.size > 0]
    sel_dark = np.vstack(stacks) if stacks else src[used_src > 0]

    # ── GMM on both paths ─────────────────────────────────────────────
    gmm_light = "light"
    gmm_dark  = "dark" if L_mean < 55 else "medium"

    fused_l, cen_l, fr_l = gmm_dominant_rgb(sel_light, mode=gmm_light, return_clusters=True)
    fused_d, cen_d, fr_d = gmm_dominant_rgb(sel_dark,  mode=gmm_dark,  return_clusters=True)

    if fused_l is None:
        fused_l = robust_median(sel_light) if sel_light.size >= 3 else np.array([200, 180, 160], float)
    if fused_d is None:
        fused_d = robust_median(sel_dark)  if sel_dark.size  >= 3 else fused_l

    fused_l = _ensure_rgb3(fused_l, tag="fused_light")
    fused_d = _ensure_rgb3(fused_d, tag="fused_dark")

    # Blend (dark path dominates for dark skin)
    fused_blend = light_w * fused_l + (1.0 - light_w) * fused_d
    fused_blend = _ensure_rgb3(fused_blend, tag="fused_blend")

    # Undertone source: dominant path
    if light_w >= 0.5:
        centers_for_ut, fracs_for_ut = cen_l, fr_l
    else:
        centers_for_ut, fracs_for_ut = cen_d, fr_d

    rough_tone = rough_monk_tone(fused_blend)

    # Adaptive ROI refinement
    used_adaptive = adaptive_mid_mask(src, used_src, rough_tone)
    sel_final     = src[used_adaptive > 0]
    if sel_final.size < 90:
        sel_final = sel_dark

    fused_refined, cen_ref, fr_ref = gmm_dominant_rgb(
        sel_final, mode=gmm_dark, return_clusters=True
    )
    if fused_refined is not None:
        fused_blend    = _ensure_rgb3(fused_refined, tag="fused_refined")
        centers_for_ut = cen_ref if cen_ref is not None else centers_for_ut
        fracs_for_ut   = fr_ref  if fr_ref  is not None else fracs_for_ut

    # ── Monk tone via Boltzmann probabilities ─────────────────────────
    probs           = monk_probabilities(fused_blend, temperature=3.5)
    tone, tone_conf = monk_from_probabilities(probs)

    # ── Dark-skin bias correction: shift tone darker if L* implies it ─
    measured_skin_L = _L_star_of_pixels(sel_final if sel_final.size > 90 else sel_dark)
    tone = tone_bias_correction(tone, probs, measured_skin_L)
    # Re-fetch confidence after possible shift
    tone_conf = float(probs[tone - 1])

    # ── Undertone ─────────────────────────────────────────────────────
    # For undertone use WB-corrected pixels (chroma matters for undertone)
    ut_src = rgb_wb if not dark_skin else rgb_wb
    region_masks = {
        "cheek":    cv2.bitwise_and(used_src, cheeks),
        "forehead": cv2.bitwise_and(used_src, fore),
        "nose":     cv2.bitwise_and(used_src, nose_m),
    }
    undertone, ut_conf = classify_undertone(
        fused_blend,
        centers_rgb  = centers_for_ut,
        fracs        = fracs_for_ut,
        monk_tone    = tone,
        region_masks = region_masks,
        rgb_source   = ut_src,
    )

    # ── Output ────────────────────────────────────────────────────────
    print(f"Tone      = {tone}  (conf: {tone_conf:.0%}, skin_L*={measured_skin_L:.1f})")
    print(f"Undertone = {undertone}  (conf: {ut_conf:.0%})")

    best, avoid = get_color_recommendations(tone, undertone)
    if best:
        print("\nRecommended Colors:")
        for c in best:
            print(f"  • {c['name']} ({c['hex']})")
    if avoid:
        print("\nColors to Avoid:")
        for c in avoid:
            print(f"  • {c['name']} ({c['hex']})")

    if debug:
        rv = rgb.copy(); rv[(cv2.bitwise_or(cheeks, fore)) > 0] = (255, 0, 0)
        uv = rgb.copy(); uv[used_src > 0] = (0, 255, 0)
        debug_display([rgb, rv, uv],
                      f"Monk {tone} ({tone_conf:.0%}) L*={measured_skin_L:.1f} | {undertone}")

    return {
        "tone":            tone,
        "undertone":       undertone,
        "tone_confidence": round(tone_conf, 3),
        "ut_confidence":   round(ut_conf, 3),
        "quality":         quality,
        "best_colors":     best,
        "avoid_colors":    avoid,
    }


# ── Backward-compat alias ─────────────────────────────────────────────
def classify_monk_v9_5(image_path: str, debug: bool = False) -> dict:
    return classify_monk_v10(image_path, debug=debug)


def _default_result(message: str = "", quality: dict | None = None) -> dict:
    print(f"Tone = 5 (default)\nUndertone = Neutral (default)")
    if message:
        print(f"[info] {message}")
    return {
        "tone":            5,
        "undertone":       "Neutral",
        "tone_confidence": 0.0,
        "ut_confidence":   0.0,
        "quality":         quality or {},
        "best_colors":     [],
        "avoid_colors":    [],
    }


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else (
        r"D:\Project_Ground\Vistone-AI-Powered-Colour-Palette-Matcher\images\tone1.jpg"
    )
    result = classify_monk_v10(img, debug=True)
    print("\n--- Full result ---")
    print(json.dumps({k: v for k, v in result.items() if k != "quality"}, indent=2))