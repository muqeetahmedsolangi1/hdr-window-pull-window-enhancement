"""Pixel-precise boundary refinement of a coarse semantic label map, using MobileSAM.

SegFormer gives us the CLASS of every region but its masks are computed at 1/4
resolution and upsampled, so the boundaries are blocky — mullions, window frames and
curtain edges come out jagged. MobileSAM (Segment Anything, lightweight) produces
pixel-precise, edge-snapped masks but is CLASS-AGNOSTIC (it does not know what a region
is). We combine the two: run SAM in "segment everything" mode on the well-exposed fused
image, then give each SAM segment the MAJORITY SegFormer class inside it. The result is
a label map with SAM-crisp boundaries and SegFormer semantics — curtains, glass and
frame separate cleanly.

Weights (~40 MB) download automatically on first use via ultralytics. If ultralytics /
the weights are unavailable, refine_label_map() returns the coarse map unchanged so the
pipeline still runs (just with SegFormer's coarser boundaries).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import cv2
import numpy as np

_sam = None
_MODEL = os.environ.get("HDR_SAM_MODEL", "mobile_sam.pt")


def _load():
    global _sam
    if _sam is None:
        from ultralytics import SAM
        _sam = SAM(_MODEL)
        print(f" * SAM refinement model: {_MODEL}")
    return _sam


def available():
    try:
        _load()
        return True
    except Exception as e:                       # pragma: no cover
        print(f" * SAM unavailable ({e}); using SegFormer boundaries")
        return False


def refine_binary(bgr, coarse_bool, sam_width=1024, min_area_frac=0.0008):
    """Refine ONE binary class mask (e.g. window, or curtain) to SAM-precise boundaries.

    Much faster than "segment everything": we take each connected component of the coarse
    mask, prompt SAM with that component's bounding box (a single forward pass decodes all
    the boxes at once), and OR the returned pixel-precise masks. Only the region we care
    about is segmented, not the whole scene. Returns a float32 [0,1] mask; falls back to
    the coarse mask (as float) if SAM is unavailable or finds nothing.
    """
    coarse_bool = coarse_bool.astype(bool)
    if not coarse_bool.any():
        return coarse_bool.astype(np.float32)
    try:
        sam = _load()
    except Exception as e:
        print(f" * SAM unavailable ({e}); using SegFormer boundaries")
        return coarse_bool.astype(np.float32)

    h, w = bgr.shape[:2]
    scale = min(1.0, sam_width / float(w))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
    cb = cv2.resize(coarse_bool.astype(np.uint8), (sw, sh),
                    interpolation=cv2.INTER_NEAREST)
    # CLOSE the coarse mask first so a curtain SegFormer split into an upper strip + a few
    # lower pixels becomes ONE component (otherwise each piece is prompted separately).
    ck = max(3, int(sh * 0.03) | 1)
    cb = cv2.morphologyEx(cb, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cb)
    boxes, boxgate = [], np.zeros((sh, sw), np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area_frac * sw * sh:
            continue
        x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        ww, hh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        pad = int(0.06 * max(ww, hh))
        # EXTEND the box DOWNWARD: curtains hang, and SegFormer often misses the lower
        # tail (occluded by furniture / against the dark sill), so its bbox stops short.
        # A box prompt returns the OBJECT inside the box, not the whole box — so giving
        # SAM extra room below lets it complete the drape WITHOUT grabbing the sill/floor
        # (those are different objects). Lateral bounds stay tight = no sideways runaway.
        downext = int(0.6 * hh)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1 = min(sw, x + ww + pad)
        y1 = min(sh, y + hh + downext)
        boxes.append([x0, y0, x1, y1])
        boxgate[y0:y1, x0:x1] = 1
    if not boxes:
        return coarse_bool.astype(np.float32)
    try:
        res = sam(small, bboxes=boxes, verbose=False)
    except Exception as e:                       # pragma: no cover
        print(f" * SAM inference failed ({e}); using SegFormer boundaries")
        return coarse_bool.astype(np.float32)
    if not res or res[0].masks is None:
        return coarse_bool.astype(np.float32)
    m = res[0].masks.data.cpu().numpy().astype(bool).any(axis=0)   # OR all box masks
    m = m & (boxgate > 0)
    out = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.float32) if out.any() else coarse_bool.astype(np.float32)


def refine_label_map(bgr, coarse, sam_width=1024, min_area=40):
    """Refine a coarse ADE20K label map to SAM-precise boundaries.

    bgr    : uint8 BGR image the labels correspond to (use the fused, well-exposed one).
    coarse : HxW int label map (SegFormer argmax), same size as bgr.
    Returns (refined_labels, seg_ids):
      refined_labels : HxW uint8 label map with SAM-crisp boundaries + SegFormer classes.
      seg_ids        : HxW int32 map giving each pixel its SAM SEGMENT index (1..N, 0 =
                       none). Lets the caller classify individual exterior segments (a
                       pool, the sky, a tree clump) by colour for the outdoor breakdown.
    Falls back to (coarse, zeros) if SAM is unavailable.

    SAM runs on a width-capped copy for speed (masks are class-agnostic shapes, so the
    small-resolution boundary upsamples cleanly). Segments are painted largest-first so
    smaller, finer segments (a mullion, a lamp) win over the big region they sit inside.
    """
    zeros = np.zeros(coarse.shape[:2], np.int32)
    try:
        sam = _load()
    except Exception as e:
        print(f" * SAM unavailable ({e}); using SegFormer boundaries")
        return coarse, zeros

    h, w = bgr.shape[:2]
    scale = min(1.0, sam_width / float(w))
    small = (cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr)
    try:
        res = sam(small, verbose=False)
    except Exception as e:                       # pragma: no cover
        print(f" * SAM inference failed ({e}); using SegFormer boundaries")
        return coarse, zeros
    if not res or res[0].masks is None:
        return coarse, zeros
    masks = res[0].masks.data.cpu().numpy().astype(bool)      # N x h' x w'

    coarse_small = cv2.resize(coarse.astype(np.int32),
                              (small.shape[1], small.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    out = coarse_small.copy()
    ids = np.zeros(coarse_small.shape, np.int32)
    for rank, i in enumerate(sorted(range(len(masks)),
                                    key=lambda j: -int(masks[j].sum())), start=1):
        m = masks[i]
        if int(m.sum()) < min_area:
            continue
        vals, cnts = np.unique(coarse_small[m], return_counts=True)
        out[m] = int(vals[np.argmax(cnts)])                  # majority SegFormer class
        ids[m] = rank                                        # SAM segment id (finest on top)
    refined = cv2.resize(out.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    seg_ids = cv2.resize(ids, (w, h), interpolation=cv2.INTER_NEAREST)
    return refined, seg_ids
