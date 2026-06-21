"""
skin_tone.py  —  Vistone v12.0
================================
Classifies a facial photo on the Google Monk Skin Tone (MST) scale (1–10)
and detects undertone (Warm / Cool / Neutral).

v12.0 key architectural changes over v11.0
-------------------------------------------
  • TWO-STAGE L*-ANCHORED CLASSIFICATION:
      Stage 1: Measure median skin L* → find closest Monk by L* only
      Stage 2: Fine-tune with ΔE (CIE2000) within ±2 Monk window
      WHY: L* is stable to WB errors; a*/b* can drift from illuminant shift.
           Warm indoor light shifts Monk 1 → predicted 3 with pure ΔE.
           L* anchor prevents this cross-group misclassification entirely.

  • IMPROVED SKIN REGION DETECTION:
      • Tight upper-cheek polygon (removes lower jaw/beard zone)
      • YCrCb + adaptive LAB combined skin pixel filter
      • Relative outlier removal: pixels > 35% darker than face median excluded
      • Distance-transform inner erosion keeps only safe central pixels

  • LIGHT SKIN FIX (Monks 1-4):
      • topL_mask (top 20% brightest) REMOVED — was biasing toward Monk 1
      • Now uses 25th–70th percentile of skin pixels (representative middle band)
      • This avoids specular highlights pulling toward lighter classification

  • DARK SKIN FIX (Monks 7-10):
      • CLAHE disabled for L_mean < 45 (prevents L* inflation of 8-12 units)
      • Gray-world WB skipped; sclera WB capped at ±15%
      • GMM dark L-band: (6, 58) covers Monk 10 (L*=14.6)
      • Dark path reads from RAW pixels (pre-WB) for chromaticity accuracy

  • FACIAL HAIR AVOIDANCE:
      • Per-pixel: exclude pixels with S > 50 AND much darker than face median
      • YCrCb Cr/Cb gating rejects non-skin pixels (hair, eyebrows, shadows)
      • Forehead hairline: top 12% of forehead polygon excluded
"""

from __future__ import annotations
import os, cv2, json, numpy as np, mediapipe as mp
from sklearn.mixture import GaussianMixture
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from image_quality import check_image_quality, quality_warning_message


# ══════════════════════════════════════════════════════════════════════
# 1.  MONK PALETTE  (Google MST — Ellis et al. 2022, skintone.google)
# ══════════════════════════════════════════════════════════════════════
MONK_TONES_HEX = [
    "#f6ede4",   # 1  L*≈94.2
    "#f3e7db",   # 2  L*≈92.3
    "#f7ead0",   # 3  L*≈93.1
    "#eadaba",   # 4  L*≈87.6
    "#d7bd96",   # 5  L*≈77.9
    "#a07e56",   # 6  L*≈55.1
    "#825c43",   # 7  L*≈42.5
    "#604134",   # 8  L*≈30.7
    "#3a312a",   # 9  L*≈21.1
    "#292420",   # 10 L*≈14.6
]

def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _to_lab_d65(rgb_01: tuple | np.ndarray) -> LabColor:
    r, g, b = (float(x) for x in rgb_01)
    return convert_color(sRGBColor(r, g, b), LabColor, target_illuminant="d65")

MONK_TONES_LAB_D65 = [_to_lab_d65(tuple(x/255.0 for x in _hex_to_rgb(h))) for h in MONK_TONES_HEX]
MONK_L_D65 = np.array([lab.lab_l for lab in MONK_TONES_LAB_D65])
# [94.21, 92.27, 93.09, 87.57, 77.90, 55.14, 42.47, 30.68, 21.07, 14.61]


# ══════════════════════════════════════════════════════════════════════
# 2.  UNDERTONE REFERENCE TABLE  (tone-stratified, D65)
#
#  IMPORTANT: L* values here reflect REAL MEASURED skin L* in photos,
#  NOT the hex palette L* (which is the pure reference swatch).
#  Real skin pixels are darker than the swatch due to lighting/texture.
#  Mismatched L* inflates all ΔE equally → gap collapses → forced Neutral.
#
#  Warm:  higher b* (yellow-orange), moderate a*
#  Cool:  lower b* (blue-pink), low a*, sometimes negative b*
#  Neutral: between warm and cool
# ══════════════════════════════════════════════════════════════════════
UNDERTONE_REFS = {
    # Monks 1-3: measured skin L* ~82-88 (not 91 from palette hex)
    (1, 3): {
        "warm":    LabColor(85.0,  8.0, 16.0),   # golden/peachy
        "cool":    LabColor(85.0,  9.0, -2.0),   # rosy/pink-blue
        "neutral": LabColor(85.0,  8.5,  7.0),   # between warm and cool
    },
    # Monks 4-5: measured skin L* ~65-78
    (4, 5): {
        "warm":    LabColor(72.0, 12.0, 20.0),   # golden beige
        "cool":    LabColor(72.0,  7.0,  1.0),   # ashy/olive-grey
        "neutral": LabColor(72.0,  9.5, 10.5),   # balanced beige
    },
    # Monks 6-7: measured skin L* ~38-52
    (6, 7): {
        "warm":    LabColor(45.0, 14.0, 21.0),   # caramel/warm brown
        "cool":    LabColor(45.0,  8.0,  3.0),   # cool brown (muted)
        "neutral": LabColor(45.0, 11.0, 12.0),   # medium brown
    },
    # Monks 8-10: measured skin L* ~18-32
    (8, 10): {
        "warm":    LabColor(24.0, 10.5, 14.0),   # warm deep brown
        "cool":    LabColor(24.0,  5.0,  0.0),   # cool/ashy deep
        "neutral": LabColor(24.0,  7.5,  7.0),   # neutral deep
    },
}

def _get_undertone_refs(monk_tone: int) -> dict:
    for (lo, hi), refs in UNDERTONE_REFS.items():
        if lo <= monk_tone <= hi:
            return refs
    return UNDERTONE_REFS[(6, 7)]


# ══════════════════════════════════════════════════════════════════════
# 3.  JSON COLOR RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════
def _load_color_json():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "monk_skin_tone_color_recommendations.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] color JSON: {e}")
        return None

COLOR_DATA = _load_color_json()

def get_color_recommendations(tone: int, undertone: str):
    if not COLOR_DATA:
        return [], []
    block = COLOR_DATA.get(f"tone_{tone}", {})
    if not isinstance(block, dict):
        return [], []
    ut    = (undertone or "").lower()
    data  = block.get(ut) or block.get("neutral") or block.get("default")
    if not data:
        return [], []
    best  = [{"name": c.get("name",""), "hex": c.get("hex","")} for c in (data.get("best")  or [])][:3]
    avoid = [{"name": c.get("name",""), "hex": c.get("hex","")} for c in (data.get("avoid") or [])][:3]
    return best, avoid


# ══════════════════════════════════════════════════════════════════════
# 4.  MEDIAPIPE
# ══════════════════════════════════════════════════════════════════════
mp_face_mesh = mp.solutions.face_mesh


# ══════════════════════════════════════════════════════════════════════
# 5.  LANDMARK GROUPS  (v12: tighter upper-cheek, no beard zone)
# ══════════════════════════════════════════════════════════════════════
# Upper cheek: landmarks in malar (cheekbone) zone only — above jaw/beard area
# Removed from old set: 58, 172, 136, 150, 176 (L) / 288, 397, 365, 379, 400 (R)
#   — those were on the lower jaw / beard zone
UPPER_LEFT_CHEEK  = [234, 93, 132, 116, 117, 118, 119, 120, 121, 128, 126, 142, 36]
UPPER_RIGHT_CHEEK = [454, 323, 361, 345, 346, 347, 348, 349, 350, 357, 355, 371, 266]

FOREHEAD_POLY  = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397]
NOSE_POLY      = [6, 197, 195, 5, 4, 45, 220, 218, 237, 1]
LEFT_EYE_RING  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LEFT_EYE_FULL  = LEFT_EYE_RING  + [130, 247, 30, 29, 27]
RIGHT_EYE_FULL = RIGHT_EYE_RING + [359, 467, 260, 259, 257]
MOUTH_OUT      = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]

def _pts(img, idxs, lm) -> np.ndarray:
    h, w = img.shape[:2]
    return np.array([(int(lm[i].x*w), int(lm[i].y*h)) for i in idxs], dtype=np.int32)

def poly_mask(shape, polys) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillConvexPoly(m, p, 255)
    return m


# ══════════════════════════════════════════════════════════════════════
# 6.  FACE DETECTION
# ══════════════════════════════════════════════════════════════════════
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROTO    = os.path.join(_BASE_DIR, "deploy.prototxt")
_MODEL    = os.path.join(_BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
_dnn_net  = None

def _load_dnn_net():
    global _dnn_net
    if _dnn_net is None and os.path.isfile(_PROTO) and os.path.isfile(_MODEL):
        try:   _dnn_net = cv2.dnn.readNetFromCaffe(_PROTO, _MODEL)
        except Exception as e: print(f"[warn] DNN: {e}")
    return _dnn_net

def fast_face_check(rgb: np.ndarray, thr: float = 0.50) -> bool:
    net = _load_dnn_net()
    if net is None: return True
    blob = cv2.dnn.blobFromImage(cv2.resize(rgb,(300,300)), 1.0, (300,300), (104,177,123))
    net.setInput(blob)
    dets = net.forward()
    return any(float(dets[0,0,i,2]) >= thr for i in range(dets.shape[2]))

def detect_mesh(rgb: np.ndarray):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5) as fm:
        res = fm.process(rgb)
    if not res.multi_face_landmarks: return None, None
    lm = res.multi_face_landmarks[0].landmark
    h, w = rgb.shape[:2]
    xs = np.array([int(p.x*w) for p in lm])
    ys = np.array([int(p.y*h) for p in lm])
    return lm, (xs.min(), ys.min(), xs.max(), ys.max())

def auto_pad(rgb, bbox, debug, save_dir, name):
    h, w = rgb.shape[:2]
    x1,y1,x2,y2 = bbox
    if (y2-y1)/max(1,h) <= 0.70: return rgb, False
    pad = int((y2-y1)*0.35)
    p   = cv2.copyMakeBorder(rgb, pad, pad//3, pad//3, pad//3,
                              cv2.BORDER_CONSTANT, value=(0,0,0))
    if debug: cv2.imwrite(os.path.join(save_dir, f"pad_{name}.png"),
                          cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
    return p, True


# ══════════════════════════════════════════════════════════════════════
# 7.  WHITE BALANCE  (conservative — capped multipliers)
# ══════════════════════════════════════════════════════════════════════
def sclera_wb(rgb: np.ndarray, eye_mask: np.ndarray,
               max_shift: float = 0.15) -> np.ndarray | None:
    """
    Sclera-based white balance. Multipliers capped to ±max_shift
    to prevent overcorrection on dark-skinned faces.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S, V = hsv[...,1], hsv[...,2]
    scl = (eye_mask > 0) & (S < 60) & (V > 120)
    if np.count_nonzero(scl) < 60: return None
    sel = rgb[scl].astype(float)
    mb, mg, mr = sel[:,0].mean()+1e-6, sel[:,1].mean()+1e-6, sel[:,2].mean()+1e-6
    gray = (mb+mg+mr)/3.0
    f = rgb.astype(np.float32)
    f[...,0] *= float(np.clip(gray/mb, 1-max_shift, 1+max_shift))
    f[...,1] *= float(np.clip(gray/mg, 1-max_shift, 1+max_shift))
    f[...,2] *= float(np.clip(gray/mr, 1-max_shift, 1+max_shift))
    return np.clip(f, 0, 255).astype(np.uint8)

def shades_of_gray_wb(rgb: np.ndarray, p: int = 6,
                       max_shift: float = 0.15) -> np.ndarray:
    """Shades-of-gray WB with capped multipliers."""
    f   = rgb.astype(np.float32) / 255.0
    eps = 1e-6
    rn  = np.power(np.mean(np.power(np.clip(f[...,2],eps,1), p)), 1/p) + eps
    gn  = np.power(np.mean(np.power(np.clip(f[...,1],eps,1), p)), 1/p) + eps
    bn  = np.power(np.mean(np.power(np.clip(f[...,0],eps,1), p)), 1/p) + eps
    gray = (rn+gn+bn)/3.0
    f[...,2] *= float(np.clip(gray/rn, 1-max_shift, 1+max_shift))
    f[...,1] *= float(np.clip(gray/gn, 1-max_shift, 1+max_shift))
    f[...,0] *= float(np.clip(gray/bn, 1-max_shift, 1+max_shift))
    return np.clip(f*255, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
# 8.  SKIN PIXEL DETECTION  (multi-channel, adaptive)
# ══════════════════════════════════════════════════════════════════════
def ycrcb_skin_mask(rgb: np.ndarray, dark_skin: bool = False) -> np.ndarray:
    """
    YCrCb skin detection. More robust than HSV alone.
    Dark skin uses wider Cr/Cb range to avoid under-segmentation.
    """
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    Cr = ycrcb[...,1].astype(int)
    Cb = ycrcb[...,2].astype(int)
    if dark_skin:
        mask = ((Cr >= 120) & (Cr <= 185) & (Cb >= 70) & (Cb <= 145))
    else:
        mask = ((Cr >= 128) & (Cr <= 180) & (Cb >= 72) & (Cb <= 140))
    return mask.astype(np.uint8) * 255


def remove_hair_and_shadows(rgb: np.ndarray, mask: np.ndarray,
                              face_median_L: float) -> np.ndarray:
    """
    Per-pixel facial hair and shadow removal using relative L* threshold.
    Pixels significantly darker than the face median are likely beard/hair/shadows.
    Threshold is relative so it works across all skin tones.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L   = lab[...,0].astype(float) * (100.0 / 255.0)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S   = hsv[...,1].astype(int)

    # Threshold: pixels below (face_median_L * 0.58) OR
    # very dark + saturated (classic beard signature)
    L_floor    = max(8.0, face_median_L * 0.58)
    hair_dark  = (L < L_floor)
    hair_beard = (S > 45) & (L < max(12.0, face_median_L * 0.50))
    hair_mask  = (hair_dark | hair_beard).astype(np.uint8) * 255

    # Also remove specular highlights (very bright, very low saturation)
    spec = ((S < 35) & (L > min(97.0, face_median_L + 30.0))).astype(np.uint8) * 255

    remove = cv2.bitwise_or(hair_mask, spec)
    clean  = cv2.bitwise_and(mask, cv2.bitwise_not(remove))
    # Morphological close to fill small gaps
    return cv2.morphologyEx(clean, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))


def percentile_L_mask(rgb: np.ndarray, mask: np.ndarray,
                       lo_pct: float, hi_pct: float) -> np.ndarray:
    """Keep only pixels in the [lo_pct, hi_pct] L* percentile range."""
    lab  = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L    = lab[...,0]
    vals = L[mask > 0]
    if vals.size < 50: return mask
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    keep   = ((L >= lo) & (L <= hi)).astype(np.uint8) * 255
    return cv2.bitwise_and(mask, keep)


def inner_erode(mask: np.ndarray, px: int) -> np.ndarray:
    """Keep only pixels > px from the mask boundary (distance transform)."""
    m    = (mask > 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return (dist > px).astype(np.uint8) * 255


def get_clean_skin_pixels(rgb: np.ndarray, base_mask: np.ndarray,
                            face_median_L: float,
                            dark_skin: bool = False,
                            lo_pct: float = 20.0,
                            hi_pct: float = 80.0,
                            min_pixels: int = 60) -> np.ndarray | None:
    """
    Full skin pixel extraction pipeline:
      1. YCrCb skin filter
      2. Relative hair/shadow removal
      3. Percentile L* range (avoids specular and deep shadows)
      4. Returns Nx3 uint8 array or None
    """
    m = cv2.bitwise_and(base_mask, ycrcb_skin_mask(rgb, dark_skin=dark_skin))
    m = remove_hair_and_shadows(rgb, m, face_median_L)
    m = percentile_L_mask(rgb, m, lo_pct, hi_pct)
    px = rgb[m > 0]
    if px.shape[0] < min_pixels:
        # Fallback: just percentile from base mask
        m2 = percentile_L_mask(rgb, base_mask, lo_pct, hi_pct)
        px = rgb[m2 > 0]
    return px if px.shape[0] >= min_pixels else None


# ══════════════════════════════════════════════════════════════════════
# 9.  FACE REGION MASKS (v12: safer polygons + forehead hairline clip)
# ══════════════════════════════════════════════════════════════════════
def cheek_masks(img, lm):
    """Upper cheek ROIs (beard-safe upper malar zone)."""
    lp = _pts(img, UPPER_LEFT_CHEEK,  lm)
    rp = _pts(img, UPPER_RIGHT_CHEEK, lm)
    ml = np.zeros(img.shape[:2], np.uint8)
    mr = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(ml, lp, 255)
    cv2.fillConvexPoly(mr, rp, 255)
    return ml, mr

def forehead_mask(img, lm, skin_L_mean: float = 60.0) -> np.ndarray:
    """
    Forehead polygon with:
      • Adaptive L* floor (dark skin)
      • Top 12% clipped to avoid hairline pixels
    """
    pts = _pts(img, FOREHEAD_POLY, lm)
    m   = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, pts, 255)
    # Clip hairline: remove top 12% of the polygon
    ys  = pts[:,1]
    y_top, y_bot = ys.min(), ys.max()
    hairline_clip = int(y_top + (y_bot - y_top) * 0.12)
    m[:hairline_clip, :] = 0
    # Erode for safety
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8)))
    return m

def nose_mask_safe(img, lm) -> np.ndarray:
    """Nose mask with moderate erosion."""
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, _pts(img, NOSE_POLY, lm), 255)
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6)))

def eyes_mask(img, lm) -> np.ndarray:
    m = np.zeros(img.shape[:2], np.uint8)
    for idxs in (LEFT_EYE_RING, RIGHT_EYE_RING):
        cv2.fillConvexPoly(m, _pts(img, idxs, lm), 255)
    return m

def eyes_exclusion(img, lm, px: int = 16) -> np.ndarray:
    m = np.zeros(img.shape[:2], np.uint8)
    for idxs in (LEFT_EYE_FULL, RIGHT_EYE_FULL):
        cv2.fillConvexPoly(m, _pts(img, idxs, lm), 255)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(px,px)))

def mouth_exclusion(img, lm, px: int = 16) -> np.ndarray:
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, _pts(img, MOUTH_OUT, lm), 255)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(px,px)))


# ══════════════════════════════════════════════════════════════════════
# 10. ROBUST COLOR ESTIMATION
# ══════════════════════════════════════════════════════════════════════
def robust_median(px: np.ndarray) -> np.ndarray | None:
    if px is None or px.size == 0: return None
    px  = px.astype(float)
    med = np.median(px, 0)
    d   = np.linalg.norm(px - med, axis=1)
    k   = px[d <= np.percentile(d, 80)]
    return np.median(k if k.size >= 50 else px, 0)

def _ensure3(v, fallback=None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.size < 3 or not np.all(np.isfinite(v[:3])):
        return np.asarray(fallback if fallback is not None else [128,128,128], np.float32)[:3]
    return v[:3]

def gmm_dominant(pixels: np.ndarray) -> np.ndarray | None:
    """BIC-optimal GMM (n=1–4), returns dominant cluster centroid."""
    import warnings
    if pixels is None or pixels.shape[0] < 80: return None
    X = pixels.astype(np.float32)
    X = X[np.all(np.isfinite(X), 1)]
    if X.shape[0] < 80: return None
    # Cap n_components to the number of unique rows to avoid ConvergenceWarning
    n_unique = len(np.unique(X, axis=0))
    max_n    = min(4, max(1, n_unique))
    best_bic, best = np.inf, None
    for n in range(1, max_n + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = GaussianMixture(n, covariance_type="full",
                                    random_state=0, reg_covar=1e-5, n_init=1)
                g.fit(X); b = g.bic(X)
            if b < best_bic: best_bic, best = b, g
        except: continue
    if best is None: return robust_median(X)
    labels = best.predict(X)
    frac   = np.bincount(labels, minlength=best.n_components).astype(float)
    frac  /= frac.sum()
    return best.means_[int(np.argmax(frac))].astype(np.float32)


def median_L_of(px: np.ndarray) -> float:
    """Median L* (D65) of an Nx3 uint8 pixel array."""
    if px is None or px.size == 0: return 50.0
    med = np.median(px.astype(float), axis=0)
    lab = _to_lab_d65(tuple(med / 255.0))
    return float(lab.lab_l)


# ══════════════════════════════════════════════════════════════════════
# 11. L*-ANCHORED TWO-STAGE TONE CLASSIFICATION  (core fix)
# ══════════════════════════════════════════════════════════════════════
def classify_tone(skin_pixels: np.ndarray,
                   fused_rgb:   np.ndarray,
                   window: int = 2) -> tuple[int, float, float]:
    """
    Two-stage Monk tone classification with L*-weighted scoring.

    Stage 1: Median L* of skin pixels → find closest Monk by L* (illuminant-stable)
    Stage 2: Combined score = ΔE_chroma + L*_penalty within ±window
             This prevents WB-induced chroma shifts from overriding L* proximity.
             e.g. Warm light on Monk 1 raises b* to ~15.8 (looks like Monk 3 in
             pure ΔE) but L* stays ~94.5 — L* penalty correctly keeps Monk 1.

    Returns (tone 1-10, confidence 0-1, measured_L).
    """
    # Stage 1: L* anchor
    measured_L = median_L_of(skin_pixels)
    L_dists    = np.abs(MONK_L_D65 - measured_L)
    anchor_idx = int(np.argmin(L_dists))   # 0-indexed

    # Stage 2: score each candidate in ±window
    lo = max(0, anchor_idx - window)
    hi = min(9, anchor_idx + window)

    v   = _ensure3(fused_rgb)
    lab = _to_lab_d65(tuple(v / 255.0))

    scores = {}
    for idx in range(lo, hi + 1):
        dE_total = delta_e_cie2000(lab, MONK_TONES_LAB_D65[idx])
        # L* deviation penalty — weighted by how tight the L* cluster is:
        # For Monks 1-3 (L* gap ~1-5), a 2-unit L* shift matters a lot.
        # For Monks 5-8 (L* gap ~10-23), L* has more room to move.
        ref_L     = MONK_L_D65[idx]
        L_dev     = abs(measured_L - ref_L)
        # Penalty weight: inversely scaled by the neighbouring L* gap
        if idx > 0 and idx < 9:
            L_gap = (abs(MONK_L_D65[idx] - MONK_L_D65[idx-1]) +
                     abs(MONK_L_D65[idx] - MONK_L_D65[idx+1])) / 2.0
        elif idx == 0:
            L_gap = abs(MONK_L_D65[0] - MONK_L_D65[1])
        else:
            L_gap = abs(MONK_L_D65[9] - MONK_L_D65[8])
        # When L* gap between neighbours is small (tight cluster), penalise more
        L_weight  = float(np.clip(8.0 / max(L_gap, 1.0), 0.3, 2.5))
        penalty   = L_dev * L_weight
        scores[idx] = dE_total + penalty

    best_idx  = min(scores, key=scores.get)
    best_tone = best_idx + 1

    # Confidence: margin between best and second-best score
    sorted_scores = sorted(scores.values())
    if len(sorted_scores) >= 2:
        margin = sorted_scores[1] - sorted_scores[0]
        conf   = float(np.clip(margin / 5.0, 0.0, 1.0))
    else:
        conf = 0.80

    return best_tone, conf, measured_L


# ══════════════════════════════════════════════════════════════════════
# 12. UNDERTONE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════
def _ut_from_lab(lab: LabColor, monk_tone: int, margin: float = 1.0) -> str:
    """
    Classify undertone from a LAB pixel.
    Dynamically adjusts the reference L* to match the pixel's actual L*
    to avoid large L* differences inflating all ΔE equally.
    margin: minimum ΔE gap between best and second-best to avoid Neutral default.
    """
    refs = _get_undertone_refs(monk_tone)

    # Shift all refs to the pixel's actual L* so L* doesn't dominate ΔE
    pixel_L = lab.lab_l
    from colormath.color_objects import LabColor as LC
    adj = {k: LC(pixel_L, v.lab_a, v.lab_b) for k, v in refs.items()}

    d_warm = delta_e_cie2000(lab, adj["warm"])
    d_cool = delta_e_cie2000(lab, adj["cool"])
    d_neut = delta_e_cie2000(lab, adj["neutral"])
    best   = min(d_warm, d_cool, d_neut)
    second = sorted([d_warm, d_cool, d_neut])[1]
    if (second - best) < margin: return "Neutral"
    if best == d_warm:  return "Warm"
    if best == d_cool:  return "Cool"
    return "Neutral"

def classify_undertone(skin_pixels: np.ndarray,
                        monk_tone:   int) -> tuple[str, float]:
    """
    Undertone from skin pixels using multi-sample voting.
    Uses the 30th–70th percentile L* pixels (avoids shadows & highlights).
    """
    if skin_pixels is None or skin_pixels.shape[0] < 50:
        return "Neutral", 0.0

    # Get representative sample (middle band)
    lab_img   = cv2.cvtColor(skin_pixels.reshape(-1,1,3).astype(np.uint8),
                              cv2.COLOR_RGB2LAB).reshape(-1,3)
    L_vals    = lab_img[:,0].astype(float) * (100.0/255.0)
    lo, hi    = np.percentile(L_vals, [30, 70])
    mid_px    = skin_pixels[(L_vals >= lo) & (L_vals <= hi)]
    if mid_px.shape[0] < 30:
        mid_px = skin_pixels

    votes = {"Warm": 0.0, "Cool": 0.0, "Neutral": 0.0}

    # Vote 1: median color
    med = np.median(mid_px.astype(float), axis=0)
    lab = _to_lab_d65(tuple(med / 255.0))
    votes[_ut_from_lab(lab, monk_tone)] += 2.0

    # Vote 2: GMM dominant
    dom = gmm_dominant(mid_px)
    if dom is not None:
        lab_d = _to_lab_d65(tuple(_ensure3(dom) / 255.0))
        votes[_ut_from_lab(lab_d, monk_tone)] += 1.5

    # Vote 3: sample 8 random pixels from mid band
    n_sample = min(8, mid_px.shape[0])
    idx_s    = np.linspace(0, mid_px.shape[0]-1, n_sample, dtype=int)
    for px in mid_px[idx_s]:
        lab_s = _to_lab_d65(tuple(px.astype(float) / 255.0))
        votes[_ut_from_lab(lab_s, monk_tone)] += 0.3

    total = sum(votes.values())
    best  = max(votes, key=votes.get)
    conf  = float(votes[best] / total) if total > 0 else 0.0
    # Cap at 92% — 100% confidence is misleading
    return best, min(conf, 0.92)


# ══════════════════════════════════════════════════════════════════════
# 13. DEBUG
# ══════════════════════════════════════════════════════════════════════
def debug_display(imgs: list, title: str = ""):
    out = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in imgs if im is not None]
    c   = np.hstack(out)
    if c.shape[1] > 900:
        c = cv2.resize(c, None, fx=900/c.shape[1], fy=900/c.shape[1],
                       interpolation=cv2.INTER_AREA)
    cv2.imshow(title, c); cv2.waitKey(0); cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════
# 14. MAIN CLASSIFIER  v12.0
# ══════════════════════════════════════════════════════════════════════
def classify_monk_v10(image_path: str, debug: bool = False) -> dict:
    """
    Skin-tone + undertone pipeline — Vistone v12.0.
    Returns dict: tone, undertone, tone_confidence, ut_confidence,
                  quality, best_colors, avoid_colors.
    """
    # ── Load ──────────────────────────────────────────────────────────
    bgr = cv2.imread(image_path)
    if bgr is None: return _default_result("Could not read image.")
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    name = os.path.splitext(os.path.basename(image_path))[0]
    sdir = os.path.dirname(image_path)

    # ── Fast DNN check ────────────────────────────────────────────────
    if not fast_face_check(rgb):
        return _default_result("No face detected.")

    # ── FaceMesh ──────────────────────────────────────────────────────
    lm, bbox = detect_mesh(rgb)
    if lm is None: return _default_result("No face landmarks.")

    # ── Quality gate ──────────────────────────────────────────────────
    quality = check_image_quality(rgb, face_landmarks=lm)
    if not quality["is_usable"]:
        return _default_result(quality_warning_message(quality), quality=quality)

    # ── Auto-pad if face fills frame ──────────────────────────────────
    rgb_p, padded = auto_pad(rgb, bbox, debug, sdir, name)
    if padded:
        lm2, bbox2 = detect_mesh(rgb_p)
        if lm2: rgb, lm, bbox = rgb_p, lm2, bbox2

    # ── Build exclusion masks ─────────────────────────────────────────
    eye_exc   = eyes_exclusion(rgb, lm, px=16)
    mouth_exc = mouth_exclusion(rgb, lm, px=18)
    excl      = cv2.bitwise_or(eye_exc, mouth_exc)

    # ── Cheek masks (upper only — beard-safe) ─────────────────────────
    ml, mr    = cheek_masks(rgb, lm)
    ml        = cv2.bitwise_and(ml, cv2.bitwise_not(excl))
    mr        = cv2.bitwise_and(mr, cv2.bitwise_not(excl))
    cheeks    = cv2.bitwise_or(ml, mr)

    # ── Estimate skin brightness from RAW cheek pixels ───────────────
    raw_cheek_px = rgb[cheeks > 0]
    skin_L_mean  = median_L_of(raw_cheek_px) if raw_cheek_px.size > 0 else 60.0
    dark_skin    = skin_L_mean < 46.0   # Monks 7-10

    print(f"[v12] skin_L_mean={skin_L_mean:.1f}  dark_skin={dark_skin}")

    # ── CLAHE (disabled for dark skin to prevent L* inflation) ────────
    if not dark_skin:
        x1,y1,x2,y2 = bbox
        h, w = rgb.shape[:2]
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
        if x2 > x1 and y2 > y1:
            face_patch = rgb[y1:y2, x1:x2].copy()
            lab_p = cv2.cvtColor(face_patch, cv2.COLOR_RGB2LAB)
            L_ch, a_ch, b_ch = cv2.split(lab_p)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
            lab_eq = cv2.merge([clahe.apply(L_ch), a_ch, b_ch])
            rgb[y1:y2, x1:x2] = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # ── White balance (conservative) ──────────────────────────────────
    max_wb = 0.12 if dark_skin else 0.15
    eye_m  = eyes_mask(rgb, lm)
    rgb_wb = sclera_wb(rgb, eye_m, max_shift=max_wb)
    if rgb_wb is None:
        rgb_wb = shades_of_gray_wb(rgb, p=6, max_shift=max_wb) if not dark_skin else rgb.copy()

    # ── Forehead + nose secondary regions ────────────────────────────
    fore  = forehead_mask(rgb_wb, lm, skin_L_mean=skin_L_mean)
    nose  = nose_mask_safe(rgb_wb, lm)
    fore  = cv2.bitwise_and(fore, cv2.bitwise_not(excl))
    nose  = cv2.bitwise_and(nose, cv2.bitwise_not(excl))

    # ── Rebuild cheek masks on WB image ──────────────────────────────
    ml_wb, mr_wb = cheek_masks(rgb_wb, lm)
    ml_wb = cv2.bitwise_and(ml_wb, cv2.bitwise_not(excl))
    mr_wb = cv2.bitwise_and(mr_wb, cv2.bitwise_not(excl))
    cheeks_wb = cv2.bitwise_or(ml_wb, mr_wb)

    # ── Inner cheek erosion (removes edge pixels touching beard zone) ─
    inner_cheeks = inner_erode(cheeks_wb, px=10)

    # ── Choose percentile range by skin tone ──────────────────────────
    if skin_L_mean > 75:          # Light skin (Monks 1-4)
        lo_pct, hi_pct = 28.0, 72.0   # MIDDLE band — avoids highlights
    elif skin_L_mean > 45:        # Medium skin (Monks 5-7)
        lo_pct, hi_pct = 15.0, 85.0
    else:                         # Dark skin (Monks 8-10)
        lo_pct, hi_pct =  5.0, 95.0   # wide — dark skin has limited L* range

    # ── Source for colour measurement ─────────────────────────────────
    # Dark skin: read from RAW (pre-WB) to avoid WB chromaticity distortion
    src = rgb if dark_skin else rgb_wb

    # ── Extract clean skin pixels per region ─────────────────────────
    def _px(region_mask):
        return get_clean_skin_pixels(src, region_mask, skin_L_mean,
                                      dark_skin=dark_skin,
                                      lo_pct=lo_pct, hi_pct=hi_pct)

    px_inner   = _px(inner_cheeks)
    px_full_ck = _px(cheeks_wb)
    px_fore    = _px(fore)
    px_nose    = _px(nose)

    # ── Build combined pixel pool for tone classification ─────────────
    # Priority: inner cheeks (most reliable) > full cheeks > forehead > nose
    pools = [p for p in [px_inner, px_full_ck, px_fore, px_nose] if p is not None]
    all_px = np.vstack(pools) if pools else src[inner_cheeks > 0]
    if all_px.shape[0] < 60:
        all_px = src[cheeks_wb > 0]
    if all_px.shape[0] < 30:
        all_px = src.reshape(-1, 3)  # last resort: full image

    # ── Fused representative color (GMM dominant) ─────────────────────
    fused = gmm_dominant(all_px)
    if fused is None:
        fused = robust_median(all_px)
    fused = _ensure3(fused, fallback=[128,128,128])

    # ── L*-ANCHORED TWO-STAGE TONE CLASSIFICATION ─────────────────────
    tone, tone_conf, measured_L = classify_tone(all_px, fused, window=2)

    # ── Undertone ─────────────────────────────────────────────────────
    # For undertone, use WB-corrected cheek pixels (chroma matters)
    ut_pools = [p for p in [
        get_clean_skin_pixels(rgb_wb, inner_cheeks, skin_L_mean,
                               dark_skin=dark_skin, lo_pct=lo_pct, hi_pct=hi_pct),
        get_clean_skin_pixels(rgb_wb, cheeks_wb, skin_L_mean,
                               dark_skin=dark_skin, lo_pct=lo_pct, hi_pct=hi_pct),
    ] if p is not None]
    ut_px = np.vstack(ut_pools) if ut_pools else all_px
    undertone, ut_conf = classify_undertone(ut_px, tone)

    # ── Print ─────────────────────────────────────────────────────────
    print(f"Tone      = {tone}  (conf: {tone_conf:.0%}, L*={measured_L:.1f})")
    print(f"Undertone = {undertone}  (conf: {ut_conf:.0%})")

    best, avoid = get_color_recommendations(tone, undertone)
    if best:
        print("\nRecommended Colors:")
        for c in best: print(f"  * {c['name']} ({c['hex']})")
    if avoid:
        print("\nColors to Avoid:")
        for c in avoid: print(f"  * {c['name']} ({c['hex']})")

    if debug:
        vis = rgb.copy()
        vis[cheeks_wb > 0] = (0, 255, 0)
        vis[fore > 0] = (255, 165, 0)
        vis[inner_cheeks > 0] = (0, 0, 255)
        debug_display([rgb, vis], f"Monk {tone} ({tone_conf:.0%}) L*={measured_L:.1f} | {undertone}")

    return {
        "tone":            tone,
        "undertone":       undertone,
        "tone_confidence": round(tone_conf, 3),
        "ut_confidence":   round(ut_conf, 3),
        "quality":         quality,
        "best_colors":     best,
        "avoid_colors":    avoid,
    }


# ── Aliases ───────────────────────────────────────────────────────────
def classify_monk_v9_5(image_path: str, debug: bool = False) -> dict:
    return classify_monk_v10(image_path, debug=debug)


def _default_result(msg: str = "", quality: dict | None = None) -> dict:
    if msg: print(f"[info] {msg}")
    return {"tone": 5, "undertone": "Neutral",
            "tone_confidence": 0.0, "ut_confidence": 0.0,
            "quality": quality or {}, "best_colors": [], "avoid_colors": []}


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else (
        r"D:\Project_Ground\Vistone-AI-Powered-Colour-Palette-Matcher\images\tone1.jpg"
    )
    result = classify_monk_v10(img, debug=True)
    print("\n--- Result ---")
    print(json.dumps({k: v for k, v in result.items() if k != "quality"}, indent=2))