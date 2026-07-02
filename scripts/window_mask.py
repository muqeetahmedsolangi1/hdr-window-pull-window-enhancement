"""Semantic masks via SegFormer (ADE20K-pretrained).

Provides soft masks for windows (white-balance exclusion), walls+ceiling
(forced white-balance) and floors (protected from neutralization), plus
lamp/cabinet/notwindow helpers.

The model downloads automatically from Hugging Face on first use and runs
on CPU. If torch/transformers are not installed, the getters return None
and the pipeline falls back to classical luminosity logic.

(Verbatim from the pixelswift project so hdr-enhance reproduces the
first-checkbox "HDR enhance" result exactly.)
"""
import os

import cv2
import numpy as np

# ADE20K class ids
WINDOW_CLASSES = (8, 14)        # windowpane, door (glass doors)
WALLCEIL_CLASSES = (0, 5)       # wall, ceiling
FLOOR_CLASSES = (3, 28)         # floor, rug
LAMP_CLASSES = (36, 82, 85)     # lamp, light/sconce, chandelier
CABINET_CLASSES = (10, 15, 24)  # cabinet, table, shelf (warm wood surfaces)
# the window "SURROUND": everything hanging over / beside the glass that is NOT the
# outdoor view — curtains, drapes, blinds, valances. Segmented separately so the
# window-view (outdoor) work can EXCLUDE them and they instead get the normal indoor
# HDR (a curtain drawn across a window must never receive the pulled exterior view).
CURTAIN_CLASSES = (18, 63)      # curtain/drape, blind/screen

# OUTDOOR classes — used to break the exterior VIEW seen through the glass into sky /
# water / greenery / buildings / ground. The whole-image model calls a window one flat
# "windowpane"; we recover this breakdown by re-segmenting a CROP of each window (see
# outdoor_breakdown), where the model sees a full outdoor scene instead of a window.
OUTDOOR_SKY = (2,)                       # sky
OUTDOOR_WATER = (21, 26, 60, 113, 128)   # water, sea, river, waterfall, lake
OUTDOOR_GREEN = (4, 9, 17, 29, 66, 72)   # tree, grass, plant, field, flower, palm
OUTDOOR_BUILD = (1, 25, 48, 79)          # building, house, skyscraper, hovel
OUTDOOR_GROUND = (13, 46, 52, 91, 94)    # earth, sand, path, dirt, land
# things that are NOT windows but often reflect light / get mislabelled as a
# window: framed pictures/paintings, mirrors, glass tables, lamps. Excluded
# from the window mask so a sunlit picture frame or a glass table-top is never
# cropped/enhanced as a window.
NOTWINDOW_CLASSES = (22, 27, 36, 82, 85, 15)  # painting, mirror, lamps, table

# Stronger model: SegFormer-b5 (same ADE20K classes/API as b0, but far more
# accurate) correctly labels framed pictures as 'painting' and mirrors as
# 'mirror' instead of mislabelling them 'windowpane' — so they are excluded
# from window detection. Slower on CPU (~5-10s) and a bigger one-time download.
# Falls back to the small fast b0 if the big model can't be loaded. Override
# with AUTOHDR_SEG_MODEL (e.g. set it back to the b0 id for speed).
MODEL_ID = os.environ.get(
    "AUTOHDR_SEG_MODEL", "nvidia/segformer-b5-finetuned-ade-640-640")
FALLBACK_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
        for mid in (MODEL_ID, FALLBACK_ID):
            try:
                _processor = SegformerImageProcessor.from_pretrained(mid)
                _model = SegformerForSemanticSegmentation.from_pretrained(mid)
                _model.eval()
                print(f" * segmentation model: {mid}")
                break
            except Exception as e:
                print(f" * seg model {mid} failed ({e}); trying fallback")
                _model = None
        if _model is None:
            raise RuntimeError("no segmentation model could be loaded")
    return _model, _processor


def get_label_map(bgr):
    """Raw ADE20K argmax label map (HxW uint8) at full resolution, or None if
    segmentation is unavailable. This is the single source the pipeline refines with
    SAM and then slices into the per-class masks (see masks_from_label_map)."""
    try:
        import torch
        model, processor = _load()
    except Exception:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits  # 1 x classes x h/4 x w/4
    labels = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
    h, w = bgr.shape[:2]
    return cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)


def masks_from_label_map(labels):
    """Slice a (SAM-refined) ADE20K label map into the soft per-class masks the
    pipeline consumes. Entries are None when that class is absent."""
    def soft(classes):
        m = np.isin(labels, classes).astype(np.float32)
        return np.clip(m, 0, 1) if m.sum() >= 10 else None
    return {
        "window": soft(WINDOW_CLASSES),
        "wallceil": soft(WALLCEIL_CLASSES),
        "floor": soft(FLOOR_CLASSES),
        "lamp": soft(LAMP_CLASSES),
        "cabinet": soft(CABINET_CLASSES),
        "notwindow": soft(NOTWINDOW_CLASSES),
        "curtain": soft(CURTAIN_CLASSES),
    }


def get_label_masks(bgr):
    """Dict of soft float32 masks {'window','wallceil','floor','lamp',
    'cabinet','notwindow','curtain'} at full resolution, or None if segmentation is
    unavailable. Individual entries are None when that class is absent."""
    labels = get_label_map(bgr)
    if labels is None:
        return None
    return masks_from_label_map(labels)


def outdoor_breakdown(bgr, win_bool, min_area_frac=0.002, target=640):
    """Break the exterior VIEW (inside `win_bool`) into sky / water / green / build /
    ground by CROP-AND-RESEGMENT.

    The whole-image model labels a window as one flat "windowpane" because it reads the
    scene as "a window in a room". If instead we crop a single window's bounding box and
    upscale it, the model sees "an outdoor photo" and labels the sky, water, trees and
    grass directly. We do that per window component, keep only the labels that fall
    inside the window mask, and return full-res bool masks. Returns None if segmentation
    is unavailable.
    """
    try:
        import torch  # noqa: F401
        _load()
    except Exception:
        return None
    H, W = bgr.shape[:2]
    full = np.full((H, W), 255, np.uint8)          # 255 = outside any window
    n, lab, st, _ = cv2.connectedComponentsWithStats(win_bool.astype(np.uint8))
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area_frac * H * W:
            continue
        x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        ww, hh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        crop = bgr[y:y + hh, x:x + ww]
        sc = target / float(max(ww, hh))
        crop_r = (cv2.resize(crop, (max(1, int(ww * sc)), max(1, int(hh * sc))),
                             interpolation=cv2.INTER_CUBIC) if sc > 1 else crop)
        lm = get_label_map(crop_r)
        if lm is None:
            continue
        lm = cv2.resize(lm, (ww, hh), interpolation=cv2.INTER_NEAREST)
        comp = lab[y:y + hh, x:x + ww] == i
        blk = full[y:y + hh, x:x + ww]
        blk[comp] = lm[comp]
        full[y:y + hh, x:x + ww] = blk

    def mk(classes):
        m = np.isin(full, classes)
        return m if m.sum() >= 10 else None
    return {
        "sky": mk(OUTDOOR_SKY),
        "water": mk(OUTDOOR_WATER),
        "green": mk(OUTDOOR_GREEN),
        "build": mk(OUTDOOR_BUILD),
        "ground": mk(OUTDOOR_GROUND),
    }


def classify_outdoor_segments(bgr_glass, seg_ids, win_bool, min_area=150):
    """Break the exterior VIEW into sky / water / green / build using the SAM segments.

    Crop-and-resegment proved unreliable on small window views (it missed the pool and
    came out patchy). Instead we reuse the SAM segments already computed for the boundary
    refinement — the pool, the sky and each tree clump are their OWN precise SAM segment —
    and classify each segment that lies inside the window by its colour in the crisp glass
    bracket (hue / saturation / brightness + vertical position). Precise boundaries (SAM)
    + robust labels (colour). Returns dict of full-res bool masks; empty if no seg ids.
    """
    hsv = cv2.cvtColor(bgr_glass, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = (hsv[..., 0].astype(np.float32),
                  hsv[..., 1].astype(np.float32),
                  hsv[..., 2].astype(np.float32))
    hh, ww = seg_ids.shape
    yy = np.repeat(np.arange(hh)[:, None], ww, axis=1)
    out = {k: np.zeros((hh, ww), bool) for k in ("sky", "water", "green", "build")}
    for sid in np.unique(seg_ids):
        if sid == 0:
            continue
        seg = (seg_ids == sid) & win_bool
        n = int(seg.sum())
        if n < min_area:
            continue
        h_ = float(np.median(Hc[seg]))       # OpenCV hue 0..179
        s_ = float(np.median(Sc[seg]))
        v_ = float(np.median(Vc[seg]))
        yc = float(yy[seg].mean()) / hh      # 0 top .. 1 bottom
        blue = 90 <= h_ <= 140
        green = 30 <= h_ <= 89
        if v_ > 150 and (s_ < 70 or blue) and yc < 0.6:
            out["sky"][seg] = True           # bright + white/blue + upper = sky
        elif blue and s_ > 45:
            out["water"][seg] = True         # saturated blue lower down = water
        elif green and s_ > 35:
            out["green"][seg] = True         # green hue = foliage
        else:
            out["build"][seg] = True         # neutral/structured = building/other
    return {k: (v if v.sum() >= 10 else None) for k, v in out.items()}


def classify_outdoor_pixels(bgr_glass, win_bool, min_area=200):
    """Best-effort breakdown of the exterior VIEW into sky / water / green / other, by
    per-pixel colour of the crisp glass bracket inside the window.

    This is a lightweight display/verification aid (small, back-lit views are genuinely
    ambiguous, so treat it as approximate — the pull uses the whole view region anyway).
    Returns dict of full-res bool masks; entries are None when a class is basically empty.
    """
    hsv = cv2.cvtColor(bgr_glass, cv2.COLOR_BGR2HSV)
    Hc = hsv[..., 0].astype(np.float32)
    Sc = hsv[..., 1].astype(np.float32)
    Vc = hsv[..., 2].astype(np.float32)
    win = win_bool
    blue = (Hc >= 90) & (Hc <= 140)
    green = (Hc >= 30) & (Hc <= 89)
    sky = win & (Vc > 150) & ((Sc < 70) | blue)
    water = win & blue & (Sc > 60) & ~sky
    greenm = win & green & (Sc > 40) & ~sky
    other = win & ~(sky | water | greenm)

    def clean(m):
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return m.astype(bool) if m.sum() >= min_area else None
    return {"sky": clean(sky), "water": clean(water),
            "green": clean(greenm), "build": clean(other)}


def get_window_mask(bgr, feather=51):
    """Back-compat: soft window mask only."""
    masks = get_label_masks(bgr)
    return None if masks is None else masks["window"]
