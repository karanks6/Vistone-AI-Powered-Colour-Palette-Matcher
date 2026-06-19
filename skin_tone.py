import os, cv2, json, numpy as np, mediapipe as mp
from sklearn.mixture import GaussianMixture
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# ---------------- Monk palette / LAB tables ----------------
MONK_TONES = [
    "#f6ede4","#f3e7db","#f7ead0","#eadaba","#d7bd96",
    "#a07e56","#825c43","#604134","#3a312a","#292420"
]
def hex_to_rgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
MONK_TONES_LAB = [convert_color(sRGBColor(r/255,g/255,b/255), LabColor)
                   for r,g,b in [hex_to_rgb(c) for c in MONK_TONES]]
def lab_d65_from_hex(hx):
    r,g,b = hex_to_rgb(hx)
    return convert_color(sRGBColor(r/255,g/255,b/255), LabColor, target_illuminant='d65')
MONK_TONES_LAB_D65 = [lab_d65_from_hex(h) for h in MONK_TONES]

mp_face_mesh = mp.solutions.face_mesh

# ---------------- Load JSON color mapping (same folder) ----------------
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
        print(f"[warn] Could not load color JSON at {path}: {e}")
        return None

COLOR_DATA = _load_color_json()

def get_color_recommendations(tone, undertone):
    """
    Returns (best_list, avoid_list) from COLOR_DATA.
    Each list contains dicts like: {"name": "...", "hex": "#RRGGBB"}
    If missing, returns ([], []).
    """
    if not COLOR_DATA:
        return [], []
    key = f"tone_{tone}"
    block = COLOR_DATA.get(key, {})
    if not isinstance(block, dict):
        return [], []
    ut = (undertone or "").lower()
    # try exact label; fallback to generic if present
    payload = block.get(ut) or block.get("default")
    if not payload:
        return [], []
    best = payload.get("best") or []
    avoid = payload.get("avoid") or []
    # sanitize
    best = [{"name": c.get("name",""), "hex": c.get("hex","")} for c in best][:3]
    avoid = [{"name": c.get("name",""), "hex": c.get("hex","")} for c in avoid][:3]
    return best, avoid

# ---------------- White balance ----------------
def shades_of_gray_wb(rgb, p=6):
    f = rgb.astype(np.float32) / 255.0; eps = 1e-6
    rn = np.power(np.mean(np.power(f[...,2], p)), 1/p) + eps
    gn = np.power(np.mean(np.power(f[...,1], p)), 1/p) + eps
    bn = np.power(np.mean(np.power(f[...,0], p)), 1/p) + eps
    m = (rn + gn + bn) / 3.0
    f[...,2] *= (m / rn); f[...,1] *= (m / gn); f[...,0] *= (m / bn)
    return np.clip(f*255.0, 0, 255).astype(np.uint8)

def sclera_based_wb(rgb, eye_mask):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); S,V = hsv[...,1], hsv[...,2]
    scl = (eye_mask>0) & (S<60) & (V>150)
    if np.count_nonzero(scl) < 150: return None
    sel = rgb[scl]
    mb,mg,mr = sel[:,0].mean()+1e-6, sel[:,1].mean()+1e-6, sel[:,2].mean()+1e-6
    gray = (mb+mg+mr)/3.0; f = rgb.astype(np.float32)
    f[...,0] *= (gray/mb); f[...,1] *= (gray/mg); f[...,2] *= (gray/mr)
    return np.clip(f,0,255).astype(np.uint8)

def white_patch_wb_on_mask(rgb, mask):
    sel = rgb[mask > 0]
    if sel.size < 150: return rgb
    mb,mg,mr = sel[:,0].mean()+1e-6, sel[:,1].mean()+1e-6, sel[:,2].mean()+1e-6
    gray = (mb + mg + mr) / 3.0; f = rgb.astype(np.float32)
    f[...,0] *= (gray/mb); f[...,1] *= (gray/mg); f[...,2] *= (gray/mr)
    return np.clip(f, 0, 255).astype(np.uint8)

def skin_only_gray_world(rgb, mask):
    sel = rgb[mask > 0]
    if sel.size < 150: return rgb
    mb,mg,mr = sel[:,0].mean()+1e-6, sel[:,1].mean()+1e-6, sel[:,2].mean()+1e-6
    m = (mb + mg + mr) / 3.0; f = rgb.astype(np.float32)
    f[...,0] *= (m/mb); f[...,1] *= (m/mg); f[...,2] *= (m/mr)
    return np.clip(f,0,255).astype(np.uint8)

# ---------------- Mesh / padding ----------------
def detect_mesh(rgb):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.55, min_tracking_confidence=0.55) as fm:
        res = fm.process(rgb)
    if not res.multi_face_landmarks: return None, None
    lm = res.multi_face_landmarks[0].landmark; h,w = rgb.shape[:2]
    xs = np.array([int(p.x*w) for p in lm]); ys = np.array([int(p.y*h) for p in lm])
    return lm, (xs.min(), ys.min(), xs.max(), ys.max())

def auto_pad_if_needed(rgb, bbox, debug, save_dir, name):
    h,w = rgb.shape[:2]; x1,y1,x2,y2 = bbox; fh = y2 - y1
    if fh / max(1,h) <= 0.70: return rgb, False
    pad = int(fh * 0.35)
    p = cv2.copyMakeBorder(rgb, pad, pad//3, pad//3, pad//3, cv2.BORDER_CONSTANT, value=(0,0,0))
    if debug:
        cv2.imwrite(os.path.join(save_dir, f"padded_{name}.png"), cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
    return p, True

# ---------------- Landmarks ----------------
LEFT_CHEEK  = [234, 93, 132, 58, 172, 136, 150, 176]
RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 400]
FOREHEAD_POLY = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397]
LEFT_EYE_RING  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE_RING = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]
LEFT_EYE_FULL  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,130,247,30,29,27]
RIGHT_EYE_FULL = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466,359,467,260,259,257]
MOUTH_OUT = [61,146,91,181,84,17,314,405,321,375,291,308]
NOSE_POLY = [6,197,195,5,4,45,220,218,237,1]

def idxs_xy(img, idxs, lm):
    h,w = img.shape[:2]
    return np.array([(int(lm[i].x*w), int(lm[i].y*h)) for i in idxs], dtype=np.int32)

def cheek_polys(img, lm): return [idxs_xy(img, LEFT_CHEEK, lm), idxs_xy(img, RIGHT_CHEEK, lm)]

def eyes_mask(img, lm):
    l = idxs_xy(img, LEFT_EYE_RING, lm); r = idxs_xy(img, RIGHT_EYE_RING, lm)
    m = np.zeros(img.shape[:2], np.uint8); cv2.fillConvexPoly(m, l, 255); cv2.fillConvexPoly(m, r, 255)
    return m

def eyes_exclusion_mask(img, lm, px=14):
    L = idxs_xy(img, LEFT_EYE_FULL, lm); R = idxs_xy(img, RIGHT_EYE_FULL, lm)
    m = np.zeros(img.shape[:2], np.uint8); cv2.fillConvexPoly(m, L, 255); cv2.fillConvexPoly(m, R, 255)
    if px>0: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(px,px)), 1)
    return m

def mouth_mask(img, lm, px=14):
    M = idxs_xy(img, MOUTH_OUT, lm); m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, M, 255)
    if px>0: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(px,px)), 1)
    return m

def forehead_mask(img, lm, er=10):
    p = idxs_xy(img, FOREHEAD_POLY, lm); m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, p, 255)
    if er>0: m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(er,er)), 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV); lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    S,V = hsv[...,1], hsv[...,2]; L = lab[...,0]
    skin = ((S < 80) & (L > 60)).astype(np.uint8)*255
    return cv2.bitwise_and(m, skin)

def nose_mask(img, lm):
    poly = idxs_xy(img, NOSE_POLY, lm)
    m = np.zeros(img.shape[:2], np.uint8); cv2.fillConvexPoly(m, poly, 255)
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV); lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L,S,V = lab[...,0], hsv[...,1], hsv[...,2]
    keep = ((L>60)&(S<70)&(V>60)).astype(np.uint8)*255
    return cv2.bitwise_and(m, keep)

def poly_mask(shape, polys):
    m = np.zeros(shape, np.uint8)
    for p in polys: cv2.fillConvexPoly(m, p, 255)
    return m

# ---------------- Cleanup / filtering ----------------
def clean_mask(rgb, mask):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); S,V = hsv[...,1], hsv[...,2]
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    spec = ((S<40) & (V>240)).astype(np.uint8)*255
    lips1 = cv2.inRange(hsv, np.array([0,60,60]), np.array([12,255,255]))
    lips2 = cv2.inRange(hsv, np.array([165,60,60]), np.array([180,255,255]))
    lips_rgb = ((r>150)&(g<165)&(b<165)).astype(np.uint8)*255
    lips = cv2.bitwise_or(lips1, cv2.bitwise_or(lips2, lips_rgb))
    beard = ((S>55)&(V<95)).astype(np.uint8)*255
    rem = cv2.bitwise_or(spec, cv2.bitwise_or(lips, beard))
    keep = cv2.bitwise_and(mask, cv2.bitwise_not(rem))
    return cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)

def mid_mask(rgb, mask, lp=15, hp=90):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB); L = lab[...,0]
    vals = L[mask>0]
    if vals.size < 50: return mask
    lo,hi = np.percentile(vals, [lp, hp])
    mm = ((L>=lo)&(L<=hi)).astype(np.uint8)*255
    return cv2.bitwise_and(mask, mm)

# ---------------- Robust / deltaE helpers ----------------
def robust_median(px):
    if px.size == 0: return None
    med = np.median(px, 0); d = np.linalg.norm(px-med, 1)
    thr = np.percentile(d, 80); k = px[d<=thr]
    if k.size < 50: k = px
    return np.median(k, 0)

def monk_from_rgb(v):
    v = np.asarray(v).reshape(-1)[:3]  # safety
    lab = convert_color(sRGBColor(*(v/255.0).tolist()), LabColor)
    d = [delta_e_cie2000(lab, t) for t in MONK_TONES_LAB]
    i = int(np.argmin(d)); return i+1, float(d[i])

def monk_from_rgb_d65(v):
    v = np.asarray(v).reshape(-1)[:3]
    lab = convert_color(sRGBColor(*(v/255.0).tolist()), LabColor, target_illuminant='d65')
    d = [delta_e_cie2000(lab, t) for t in MONK_TONES_LAB_D65]
    i = int(np.argmin(d)); return i+1, float(d[i])

# ---------------- GMM Dominant Color (Adaptive + Safe Mode) ----------------
def _cluster_stats(centers_rgb, counts_frac, use_d65=False):
    labs=[]; tones = MONK_TONES_LAB_D65 if use_d65 else MONK_TONES_LAB
    for c in centers_rgb:
        srgb = sRGBColor(*(np.asarray(c)/255.0).reshape(-1)[:3].tolist())
        lab = convert_color(srgb, LabColor, target_illuminant=('d65' if use_d65 else None))
        labs.append(lab)
    min_dE=[]
    for lab in labs:
        d=[delta_e_cie2000(lab,t) for t in tones]
        min_dE.append(min(d))
    Ls=[lab.lab_l for lab in labs]
    return np.array(Ls,float), np.array(min_dE,float), np.array(counts_frac,float), labs

def _ensure_rgb3(v, fallback=None, tag="fused"):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim > 1: v = v.reshape(-1)
    if v.size < 3 or not np.all(np.isfinite(v[:3])):
        if fallback is not None:
            print(f"[warn] {tag} invalid shape/NaN; using fallback.")
            return np.asarray(fallback, dtype=np.float32).reshape(-1)[:3]
        print(f"[warn] {tag} invalid; returning [128,128,128].")
        return np.array([128,128,128], np.float32)
    return v[:3]

def gmm_dominant_rgb(pixels_rgb, mode='light', return_clusters=False):
    # pixels_rgb: Nx3
    if pixels_rgb is None or pixels_rgb.size < 150:
        return (None, None, None) if return_clusters else None
    X = pixels_rgb.astype(np.float32)
    X = X[np.all(np.isfinite(X), axis=1)]
    if X.shape[0] < 150:
        return (None, None, None) if return_clusters else None

    n = 3
    try:
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=0, reg_covar=1e-6)
        gmm.fit(X)
        labels = gmm.predict(X)
        counts = np.bincount(labels, minlength=n).astype(np.float32)
        frac = counts/np.maximum(1.0, counts.sum())
        centers = gmm.means_
    except Exception as e:
        print(f"[warn] GMM fit failed: {e}. Fallback median.")
        fm = robust_median(X)
        if return_clusters: return fm, None, None
        return fm

    use_d65 = (mode=='light')
    Ls, dEs, fracs, labs = _cluster_stats(centers, frac, use_d65=use_d65)

    if mode=='light': L_lo,L_hi = 72.0, 95.0
    else:             L_lo,L_hi = 45.0, 78.0

    L_penalty = np.zeros_like(Ls)
    L_penalty += np.where(Ls<L_lo, (L_lo-Ls)*0.8, 0.0)
    L_penalty += np.where(Ls>L_hi, (Ls-L_hi)*0.6, 0.0)

    tiny = fracs < 0.05
    tiny_pen = tiny.astype(np.float32)*3.0

    score = (-dEs) + (1.2*fracs) - (0.15*L_penalty) - (tiny_pen)
    idx = int(np.argmax(score))

    center = centers[idx]
    if not np.all(np.isfinite(center)):
        print("[warn] GMM center NaN; fallback median.")
        fm = robust_median(X)
        if return_clusters: return fm, centers, fracs
        return fm

    if return_clusters:
        return center.astype(np.float32), centers.astype(np.float32), fracs.astype(np.float32)
    return center.astype(np.float32)

# ---------------- Undertone (Hybrid: LAB + ΔE + Cluster Voting) ----------------
def _undertone_from_lab(lab: LabColor):
    a, b = lab.lab_a, lab.lab_b
    if a < -2 and b < 12: return "Cool"
    if a > 8 and b > 14:  return "Warm"
    return "Neutral"

def _undertone_vote_from_clusters(centers_lab, fracs):
    COOL_REF    = LabColor(70,  5, -6)
    WARM_REF    = LabColor(70, 18, 18)
    NEUTRAL_REF = LabColor(70,  6, 18)

    w_score = 0.0; c_score = 0.0; n_score = 0.0
    weights = fracs if fracs is not None else [1/len(centers_lab)]*len(centers_lab)
    for lab, f in zip(centers_lab, weights):
        d_c = delta_e_cie2000(lab, COOL_REF)
        d_w = delta_e_cie2000(lab, WARM_REF)
        d_n = delta_e_cie2000(lab, NEUTRAL_REF)
        w_score += f * (1.0/(1.0 + d_w))
        c_score += f * (1.0/(1.0 + d_c))
        n_score += f * (1.0/(1.0 + d_n))
    if max(w_score, c_score, n_score) == w_score: return "Warm"
    if max(w_score, c_score, n_score) == c_score: return "Cool"
    return "Neutral"

def classify_undertone(fused_rgb, centers_rgb=None, fracs=None, use_d65=True):
    srgb = sRGBColor(*(_ensure_rgb3(fused_rgb)/255.0).tolist())
    lab  = convert_color(srgb, LabColor, target_illuminant=('d65' if use_d65 else None))

    NEUTRAL_REF = LabColor(70, 6, 18)
    dE_neutral  = delta_e_cie2000(lab, NEUTRAL_REF)
    if dE_neutral < 6.0:
        base = "Neutral"
    else:
        base = _undertone_from_lab(lab)

    if centers_rgb is not None and fracs is not None:
        centers_lab=[]
        for c in centers_rgb:
            sr = sRGBColor(*(_ensure_rgb3(c)/255.0).tolist())
            centers_lab.append(convert_color(sr, LabColor, target_illuminant=('d65' if use_d65 else None)))
        vote = _undertone_vote_from_clusters(centers_lab, fracs)
        if base == "Neutral" and vote != "Neutral":
            return vote
        if base in ("Warm","Cool") and vote in ("Warm","Cool") and vote != base:
            return vote
    return base

# ---------------- Stats / visuals ----------------
def hsv_stats(rgb,mask):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); S,V = hsv[...,1].astype(np.float32), hsv[...,2].astype(np.float32)
    s = S[mask>0]; v = V[mask>0]
    if s.size==0: return 0.0,0.0,0.0
    return float(np.mean(s)), float(np.std(s)), float(np.std(v))

def erode_inner_cheeks_dist(mask, px=12):
    m = (mask>0).astype(np.uint8)*255; dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return (dist > px).astype(np.uint8)*255

def topL_mask(rgb, base_mask, pct=20):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB); L = lab[...,0]; vals = L[base_mask>0]
    if vals.size < 60: return base_mask
    t = np.percentile(vals, 100 - pct); bright = ((L>=t).astype(np.uint8)*255)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); S,V = hsv[...,1], hsv[...,2]
    nonspec = (~((S<30) & (V>246))).astype(np.uint8)*255
    return cv2.bitwise_and(cv2.bitwise_and(base_mask, bright), nonspec)

def debug_display(imgs, title=""):
    out = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in imgs if im is not None]
    c = np.hstack(out)
    if c.shape[1] > 900:
        c = cv2.resize(c, None, fx=900/c.shape[1], fy=900/c.shape[1], interpolation=cv2.INTER_AREA)
    cv2.imshow(title, c); cv2.waitKey(0); cv2.destroyAllWindows()

# ---------------- Main classifier (v9.5) ----------------
def classify_monk_v9_5(image_path, debug=False):
    bgr = cv2.imread(image_path)
    if bgr is None:
        print("Tone = 5"); print("Undertone = Neutral")
        return {"tone":5,"undertone":"Neutral","best_colors":[], "avoid_colors":[]}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    name = os.path.splitext(os.path.basename(image_path))[0]; save_dir = os.path.dirname(image_path)

    lm, bbox = detect_mesh(rgb)
    if lm is None:
        print("Tone = 5"); print("Undertone = Neutral")
        return {"tone":5,"undertone":"Neutral","best_colors":[], "avoid_colors":[]}
    rgb_pad, padded = auto_pad_if_needed(rgb, bbox, debug, save_dir, name)
    if padded:
        lm2, bbox2 = detect_mesh(rgb_pad)
        if lm2 is not None: rgb, lm, bbox = rgb_pad, lm2, bbox2

    polys = cheek_polys(rgb, lm)
    cheeks = poly_mask(rgb.shape[:2], polys)
    fore   = forehead_mask(rgb, lm, er=10)
    nose   = nose_mask(rgb, lm)

    eyes_exc = eyes_exclusion_mask(rgb, lm, px=14)
    mouth    = mouth_mask(rgb, lm, px=14)

    cheeks = cv2.bitwise_and(cheeks, cv2.bitwise_not(eyes_exc))
    fore   = cv2.bitwise_and(fore,   cv2.bitwise_not(eyes_exc))
    nose   = cv2.bitwise_and(nose,   cv2.bitwise_not(eyes_exc))

    # ROI used for cleaning / visualization
    roi = cv2.bitwise_or(cheeks, fore)

    # White balance
    eye_wb = eyes_mask(rgb, lm)
    rgb_wb = sclera_based_wb(rgb, eye_wb)
    if rgb_wb is None: rgb_wb = shades_of_gray_wb(rgb, p=6)

    clean = clean_mask(rgb_wb, roi)
    used  = mid_mask(rgb_wb, cv2.bitwise_and(clean, cv2.bitwise_not(mouth)), 15, 90)

    # Light gate
    lab_tmp = cv2.cvtColor(rgb_wb, cv2.COLOR_RGB2LAB); Lch = lab_tmp[...,0]
    s_mean,_,_ = hsv_stats(rgb_wb, used)
    L_vals = Lch[used>0]; L_mean = float(np.mean(L_vals)) if L_vals.size else 0.0
    is_light = (s_mean <= 16.0 and L_mean >= 84.0)

    inner = erode_inner_cheeks_dist(cheeks, 12)

    centers_for_undertone=None; fracs_for_undertone=None

    if is_light:
        # Light: inner cheeks + forehead (NO nose)
        core = cv2.bitwise_or(inner, fore)
        b1 = topL_mask(rgb_wb, core, 20)
        rgb_local = white_patch_wb_on_mask(rgb_wb, b1)
        rgb_local = skin_only_gray_world(rgb_local, core)
        b2 = topL_mask(rgb_local, core, 20)

        sel = rgb_local[b2>0]
        if sel.size < 60: sel = rgb_local[core>0]
        if sel.size < 50: sel = rgb_wb[used>0]

        fused, cen, fr = gmm_dominant_rgb(sel, mode='light', return_clusters=True)
        if fused is None:
            print("[warn] Light GMM returned None; fallback percentile.")
            fused = np.percentile(sel, 85, 0).astype(np.float32)
        fused = _ensure_rgb3(fused, fallback=np.percentile(sel,85,0), tag="fused_light")
        centers_for_undertone, fracs_for_undertone = cen, fr

        lab_f = convert_color(sRGBColor(*(fused/255.0).tolist()), LabColor, target_illuminant='d65')
        if lab_f.lab_l >= 88.0: tone = 1
        else: tone, _ = monk_from_rgb_d65(fused)

        undertone = classify_undertone(fused, centers_for_undertone, fracs_for_undertone, use_d65=True)

    else:
        # Medium/Dark: inner cheeks + nose (NO forehead)
        Lmask = cv2.bitwise_and(poly_mask(rgb.shape[:2],[polys[0]]), inner)
        Rmask = cv2.bitwise_and(poly_mask(rgb.shape[:2],[polys[1]]), inner)
        Nmask = nose

        pxL = rgb_wb[cv2.bitwise_and(used, Lmask) > 0]
        pxR = rgb_wb[cv2.bitwise_and(used, Rmask) > 0]
        pxN = rgb_wb[cv2.bitwise_and(used, Nmask) > 0]
        if pxL.size < 50: pxL = rgb_wb[cv2.bitwise_and(clean, Lmask) > 0]
        if pxR.size < 50: pxR = rgb_wb[cv2.bitwise_and(clean, Rmask) > 0]
        if pxN.size < 50: pxN = rgb_wb[cv2.bitwise_and(clean, Nmask) > 0]

        stacks = [p for p in [pxL, pxR, pxN] if p.size]
        sel = np.vstack(stacks) if len(stacks)>0 else rgb_wb[used>0]

        if sel.size < 150:
            print("[warn] Dark selection <150 px; fallback robust median.")
            fused = robust_median(sel)
            centers_for_undertone=None; fracs_for_undertone=None
        else:
            fused, cen, fr = gmm_dominant_rgb(sel, mode='dark', return_clusters=True)
            if fused is None:
                print("[warn] Dark GMM returned None; fallback robust median.")
                fused = robust_median(sel)
                centers_for_undertone=None; fracs_for_undertone=None
            else:
                centers_for_undertone, fracs_for_undertone = cen, fr

        if fused is None:
            print("[warn] Fused None after fallbacks; using mid-gray.")
            fused = np.array([128,128,128], np.float32)

        fused = _ensure_rgb3(fused, fallback=robust_median(sel), tag="fused_dark")
        tone, _ = monk_from_rgb(fused.astype(np.float32))
        undertone = classify_undertone(fused, centers_for_undertone, fracs_for_undertone, use_d65=False)

    # ----- Print (plain text) -----
    print(f"Tone = {tone}")
    print(f"Undertone = {undertone}")

    # ----- Color recommendations -----
    best, avoid = get_color_recommendations(tone, undertone)

    if best:
        print("\nRecommended Colors:")
        for c in best:
            print(f"• {c['name']} ({c['hex']})")
    if avoid:
        print("\nColors to Avoid:")
        for c in avoid:
            print(f"• {c['name']} ({c['hex']})")

    # ----- Optional debug panels -----
    if debug:
        rv = rgb.copy(); rv[ (cv2.bitwise_or(cheeks, fore)) > 0 ] = (255,0,0)
        uv = rgb.copy(); uv[ (mid_mask(rgb_wb, cv2.bitwise_and(clean, cv2.bitwise_not(mouth)), 15,90)) > 0 ] = (0,255,0)
        debug_display([rgb, rv, uv], f"Monk {tone} | {undertone}")

    return {
        "tone": tone,
        "undertone": undertone,
        "best_colors": best,
        "avoid_colors": avoid
    }

# ---------------- Run ----------------
if __name__ == "__main__":
    img = r"D:\Project_Ground\Vistone-AI-Powered-Colour-Palette-Matcher\images\tone1.jpg"
    result = classify_monk_v9_5(img, debug=True)
    