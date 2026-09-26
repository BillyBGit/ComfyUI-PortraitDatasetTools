"""
Portrait quality metrics for LoRA training-set curation.

Pure numpy/cv2 on top of an InsightFace ``FaceAnalysis`` app the caller
supplies, so the same code runs inside the ComfyUI node and from the
offline calibration script (tools/survey.py).

Everything is measured on the FACE REGION at a normalised scale, not on the
whole frame. A sharp face in front of bokeh must pass; a sharp background
behind a soft face must fail. Normalising the face to ``norm_face_px`` short
edge before measuring makes clarity/grain comparable across a 219px and a
6590px source, and -- because a 1024 head crop holds roughly a 512px face --
it also means "sharp at the scale we will actually train on".
"""

import math

import cv2
import numpy as np

# Immerkaer fast noise-variance estimate: convolve with this, take mean |.|,
# scale by sqrt(pi/2)/6. Insensitive to edges compared to plain std-dev.
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def largest_face(app, bgr):
    """Largest detected face, with the pad-and-retry fallback the rest of the
    pack uses for tight crops that defeat SCRFD's anchors. Returns
    (face, (offset_x, offset_y)) or (None, (0, 0))."""
    faces = app.get(bgr)
    if faces:
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])), (0, 0)
    h, w = bgr.shape[:2]
    m = int(max(h, w) * 0.5)
    padded = cv2.copyMakeBorder(bgr, m, m, m, m, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    faces = app.get(padded)
    if not faces:
        return None, (0, 0)
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])), (m, m)


def chroma_stats(bgr):
    """(mean chroma, circular hue spread) in Lab. B&W scans sit at chroma <3,
    tinted/sepia monochrome at 3-8 with a tight hue spread (<0.45, one hue),
    genuine colour photos of faces from ~7 upward with spread >0.5.
    Calibrated on 176 vintage + modern portraits."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128.0, lab[..., 2] - 128.0
    ch = np.sqrt(a * a + b * b)
    m = ch > 2.0
    if m.sum() > 100:
        ang = np.arctan2(b[m], a[m])
        r = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
        spread = float(math.sqrt(-2.0 * math.log(max(r, 1e-6))))
    else:
        spread = 0.0
    return round(float(ch.mean()), 2), round(spread, 3)


def is_monochrome(chroma, hue_spread, max_chroma=5.0, sepia_max_chroma=8.0, sepia_hue_spread=0.45):
    return chroma < max_chroma or (chroma < sepia_max_chroma and hue_spread < sepia_hue_spread)


def _noise_sigma(gray_f64):
    h, w = gray_f64.shape
    if h < 8 or w < 8:
        return float("nan")
    resp = cv2.filter2D(gray_f64, -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT)
    return float(math.sqrt(math.pi / 2.0) / (6.0 * (w - 2) * (h - 2)) * np.abs(resp[1:-1, 1:-1]).sum())


def _patch(gray, cx, cy, half):
    h, w = gray.shape
    x1, x2 = int(round(cx - half)), int(round(cx + half))
    y1, y2 = int(round(cy - half)), int(round(cy + half))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return gray[y1:y2, x1:x2]


def analyse(bgr, app, norm_face_px=512, region_pad=0.15):
    """Return a dict of metrics for the largest face in ``bgr`` (uint8 BGR).

    Keys (always present):
      face_found, img_w, img_h
    When a face is found:
      face_px        short edge of the face bbox in NATIVE pixels
      face_w, face_h
      face_area_pct  bbox area as % of the frame
      det_score      SCRFD confidence
      roll_deg       eye-line roll
      clarity        variance of Laplacian on the face region at norm scale
      grain          Immerkaer noise sigma (0-255 units) on flat skin patches
                     at norm scale; nan if no usable patch
      embedding      512-d ArcFace vector (numpy) for identity clustering
    """
    img_h, img_w = bgr.shape[:2]
    out = {"face_found": False, "img_w": img_w, "img_h": img_h}

    face, (ox, oy) = largest_face(app, bgr)
    if face is None:
        return out

    bbox = face.bbox.astype(np.float64).copy()
    kps = face.kps.astype(np.float64).copy()
    bbox[[0, 2]] -= ox
    bbox[[1, 3]] -= oy
    kps[:, 0] -= ox
    kps[:, 1] -= oy

    fw, fh = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
    face_px = max(1.0, min(fw, fh))
    out.update(
        face_found=True,
        face_px=int(round(face_px)),
        face_w=int(round(fw)),
        face_h=int(round(fh)),
        face_area_pct=round(100.0 * fw * fh / float(img_w * img_h), 2),
        det_score=round(float(face.det_score), 4),
        roll_deg=round(float(np.degrees(np.arctan2(kps[1, 1] - kps[0, 1], kps[1, 0] - kps[0, 0]))), 2),
        embedding=getattr(face, "normed_embedding", None),
    )

    # -- face region, padded, clipped to frame --
    px, py = fw * region_pad, fh * region_pad
    x1 = int(max(0, math.floor(bbox[0] - px)))
    y1 = int(max(0, math.floor(bbox[1] - py)))
    x2 = int(min(img_w, math.ceil(bbox[2] + px)))
    y2 = int(min(img_h, math.ceil(bbox[3] + py)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        out.update(clarity=float("nan"), grain=float("nan"))
        return out

    region = bgr[y1:y2, x1:x2]
    out["chroma"], out["hue_spread"] = chroma_stats(region)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    scale = norm_face_px / face_px
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    g64 = gray.astype(np.float64)
    out["clarity"] = round(float(cv2.Laplacian(g64, cv2.CV_64F).var()), 2)

    # -- grain on flat skin: two cheeks + forehead, in normalised coords --
    k = (kps - np.array([x1, y1])) * scale
    l_eye, r_eye, nose, l_mouth, r_mouth = k
    eye_mid = (l_eye + r_eye) / 2.0
    ipd = float(np.linalg.norm(r_eye - l_eye)) or 1.0
    half = max(6.0, 0.22 * ipd)
    # cheek: between eye and mouth corner, pushed slightly outward from the nose
    l_cheek = (l_eye + l_mouth) / 2.0 + (l_eye - nose) * 0.25
    r_cheek = (r_eye + r_mouth) / 2.0 + (r_eye - nose) * 0.25
    forehead = eye_mid - np.array([0.0, 0.75 * ipd])
    sigmas = []
    for cx, cy in (l_cheek, r_cheek, forehead):
        p = _patch(g64, cx, cy, half)
        if p is not None:
            sigmas.append(_noise_sigma(p))
    out["grain"] = round(float(np.median(sigmas)), 3) if sigmas else float("nan")

    # mean Lab a/b of the same skin patches, in NATIVE coords -- this is the
    # target SkinToneMatch aims at, and it is dataset-specific. Medians seen:
    # a 11.2 / b 14.3 on a warm-toned set, a 10.3 / b 9.2 on an East Asian set.
    # `a` barely moves between sets; `b` is the one to measure.
    lab = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.float32)
    kn = kps - np.array([x1, y1])
    le2, re2, no2, lm2, rm2 = kn
    ipd2 = float(np.linalg.norm(re2 - le2)) or 1.0
    h2 = int(max(6, 0.22 * ipd2))
    mask = np.zeros(lab.shape[:2], bool)
    for c in ((le2 + lm2) / 2 + (le2 - no2) * 0.25,
              (re2 + rm2) / 2 + (re2 - no2) * 0.25,
              (le2 + re2) / 2 - np.array([0.0, 0.75 * ipd2])):
        x, y = int(c[0]), int(c[1])
        mask[max(0, y - h2):y + h2, max(0, x - h2):x + h2] = True
    if mask.any():
        out["skin_a"] = round(float(lab[..., 1][mask].mean() - 128.0), 2)
        out["skin_b"] = round(float(lab[..., 2][mask].mean() - 128.0), 2)
    return out


def verdict(m, min_face_px, min_clarity, max_grain, hard_floor_face_px=0):
    """Classify a metrics dict. Returns (accepted, hard_reject, reasons)."""
    if not m.get("face_found"):
        return False, True, ["no_face"]
    reasons = []
    if hard_floor_face_px > 0 and m["face_px"] < hard_floor_face_px:
        return False, True, [f"face_px {m['face_px']} < hard floor {hard_floor_face_px}"]
    if m["face_px"] < min_face_px:
        reasons.append(f"face_px {m['face_px']} < {min_face_px}")
    c, g = m.get("clarity", float("nan")), m.get("grain", float("nan"))
    if not (c >= min_clarity):
        reasons.append(f"clarity {c} < {min_clarity}")
    if g == g and g > max_grain:  # nan-safe
        reasons.append(f"grain {g} > {max_grain}")
    return (len(reasons) == 0), False, reasons


def sr_short_edge(m, face_target_px, lo=1024, hi=2560, step=16):
    """Short-edge target for a super-res pass that lands the face at
    ~face_target_px, snapped to ``step`` and clamped to [lo, hi]."""
    if not m.get("face_found") or m["face_px"] <= 0:
        return lo
    short = min(m["img_w"], m["img_h"])
    want = short * face_target_px / float(m["face_px"])
    want = int(round(want / step) * step)
    return int(max(lo, min(hi, want)))
