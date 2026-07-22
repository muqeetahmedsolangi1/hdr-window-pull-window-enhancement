# =============================================================================
# SEMANTIC FUSION PIPELINE — standalone Colab script (the NEW approach)
# =============================================================================
# THE CORE IDEA: every part of the room is built from the BRACKET that shot it
# best — chosen by understanding what each part IS — and every correction value
# is MEASURED from the image against a per-class target table (profiles), never
# hardcoded.
#
# The stages (one per cell, exactly the design):
#   1. per-bracket exposure-quality maps (well-exposedness + over/under clip)
#   2. semantic segmentation (OneFormer)
#   3. best-bracket SCORE per semantic region
#   4. semantically-modulated fusion weights
#   5. Laplacian pyramid blend  -> the fused image
#   6. data-driven per-region correction toward the PROFILE target bands
#   7. verification loop (re-measure, iterate, histogram check) + output
# =============================================================================


# ============================== CELL 1 — SETUP =================================
# On a fresh Colab runtime run this line first (then comment it again):
# !pip -q install transformers
import os, glob, math
import numpy as np
import cv2
import matplotlib.pyplot as plt

W = 2000                                              # working width


def show_hist(bgr, title):
    """Image + its histogram side by side (luma + R/G/B, clip lines marked)."""
    luma = bgr.astype(np.float32).mean(2) / 255
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
    ax[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); ax[0].axis("off"); ax[0].set_title(title)
    ax[1].hist(luma.ravel(), bins=128, range=(0, 1), color="0.4", alpha=0.9, label="luma")
    for i, c in zip((2, 1, 0), ("r", "g", "b")):
        ax[1].hist((bgr[..., i].ravel() / 255.0), bins=128, range=(0, 1),
                   histtype="step", color=c, linewidth=1.1)
    ax[1].axvline(0.05, color="c", ls="--", lw=1); ax[1].axvline(0.95, color="m", ls="--", lw=1)
    ax[1].set_xlim(0, 1); ax[1].set_yticks([])
    ax[1].set_title("histogram  (cyan=shadow clip · magenta=highlight clip)")
    plt.tight_layout(); plt.show()
    print(f"{title}: blown>0.95 = {100*(luma>0.95).mean():.1f}%  |  "
          f"crushed<0.05 = {100*(luma<0.05).mean():.1f}%  |  median = {np.median(luma):.2f}")


# ============================== CELL 2 — BRACKETS -> ALIGN =====================
SOURCE = "drive"       #@param ["drive", "upload"]
SCENE = "scene1"       #@param {type:"string"}
DRIVE_BASE = "/content/drive/MyDrive/RAW/new-exposure"   #@param {type:"string"}
SELECT = "all"         #@param {type:"string"}   # "all", "0,2,4", "0-5", mixes ok
_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.tif", "*.tiff")

if SOURCE == "drive":
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    folder = os.path.join(DRIVE_BASE, SCENE)
    paths = sorted(sum([glob.glob(os.path.join(folder, e)) for e in _EXTS], []))
    if not paths:
        raise RuntimeError(f"no images in {folder} — check the SCENE name")
else:
    from google.colab import files
    print(">>> Upload the exposure BRACKETS of ONE scene:")
    up = files.upload()
    paths = sorted(up.keys())

print(f"{len(paths)} images available:")
for _i, _p in enumerate(paths):
    print(f"  [{_i}] {os.path.basename(_p)}")


def _parse_sel(sel, n):
    sel = str(sel).strip().lower()
    if sel in ("", "all"):
        return list(range(n))
    idx = []
    for part in sel.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); idx += list(range(int(a), int(b) + 1))
        elif part:
            idx.append(int(part))
    return [i for i in idx if 0 <= i < n] or list(range(n))


paths = [paths[i] for i in _parse_sel(SELECT, len(paths))]
imgs = []
for _p in paths:
    _im = cv2.imread(_p)
    if _im is None:
        continue
    if _im.shape[1] != W:
        _im = cv2.resize(_im, (W, int(_im.shape[0] * W / _im.shape[1])),
                         interpolation=cv2.INTER_AREA)
    imgs.append(_im)
imgs.sort(key=lambda im: im.mean())                   # darkest -> brightest


def _align(imgs, wwork=1000):
    """ECC-align every bracket to the middle one (exposure-robust homography)."""
    ref = len(imgs) // 2
    h, w = imgs[0].shape[:2]
    s = min(1.0, wwork / w)
    small = [cv2.equalizeHist(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None,
             fx=s, fy=s, interpolation=cv2.INTER_AREA)) for im in imgs]
    S = np.diag([s, s, 1.0])
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    out = []
    for i, im in enumerate(imgs):
        if i == ref:
            out.append(im); continue
        warp = np.eye(3, dtype=np.float32)
        try:
            cv2.findTransformECC(small[ref], small[i], warp, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
            Hm = (np.linalg.inv(S) @ warp.astype(np.float64) @ S).astype(np.float32)
            out.append(cv2.warpPerspective(im, Hm, (w, h),
                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_REPLICATE))
        except cv2.error:
            out.append(im)
    return out


aligned = _align(imgs)
H, Wd = aligned[0].shape[:2]
N = len(aligned)
print(f"{N} brackets aligned -> {Wd}x{H}")

# ---- SHOW every input bracket (darkest -> brightest) --------------------------
fig, ax = plt.subplots(1, N, figsize=(2.6 * N, 3.0))
if N == 1:
    ax = [ax]
for _j, _a in enumerate(aligned):
    ax[_j].imshow(cv2.cvtColor(_a, cv2.COLOR_BGR2RGB)); ax[_j].axis("off")
    ax[_j].set_title(f"#{_j}", fontsize=10)
fig.suptitle("INPUT BRACKETS (darkest -> brightest)", y=1.04)
plt.tight_layout(); plt.show()


# ============================== CELL 3 — STAGE 1: QUALITY MAPS =================
# Per bracket, per pixel: how close to ideal exposure, and where clipped.
# Clipping is checked PER CHANNEL — a single blown channel (a red curtain)
# counts even when the luma does not.
IMS = [a.astype(np.float32) / 255 for a in aligned]
LUS = [im.mean(2) for im in IMS]
OVER = [(im > 0.95).any(2) for im in IMS]
UNDER = [(im < 0.05).all(2) for im in IMS]
print("bracket quality (darkest -> brightest):")
for _j in range(N):
    print(f"  #{_j}: median {np.median(LUS[_j]):.2f}   "
          f"over {100 * OVER[_j].mean():4.1f}%   under {100 * UNDER[_j].mean():4.1f}%")

# ---- SHOW the quality maps: where each bracket is blown / crushed -------------
fig, ax = plt.subplots(1, N, figsize=(2.6 * N, 3.0))
if N == 1:
    ax = [ax]
for _j in range(N):
    _vis = IMS[_j][..., ::-1].copy()                  # RGB view of the bracket
    _vis[OVER[_j]] = (1.0, 0.15, 0.15)                # RED   = blown there
    _vis[UNDER[_j]] = (0.2, 0.4, 1.0)                 # BLUE  = crushed there
    ax[_j].imshow(_vis); ax[_j].axis("off")
    ax[_j].set_title(f"#{_j}  over {100*OVER[_j].mean():.0f}% / under {100*UNDER[_j].mean():.0f}%",
                     fontsize=8)
fig.suptitle("STAGE 1 — exposure quality per bracket (RED = blown · BLUE = crushed)", y=1.04)
plt.tight_layout(); plt.show()

# quick reference fusion — used ONLY as the segmentation source (it shows every
# region un-clipped, which equals the doc's clipped-region re-segmentation merge
# in a single pass) and for the before/after comparison at the end
mertens = (cv2.createMergeMertens().process(aligned) * 255).clip(0, 255).astype("uint8")


# ============================== CELL 4 — STAGE 2: SEGMENTATION =================
SEG_SOURCE = "multi"   #@param ["multi", "fused", "mid"]
# "multi" = the 5-step MULTI-BRACKET segmentation (default):
#           1. segment the 0EV (mid) bracket -> masks
#           2. flag masks overlapping clipped regions (>0.95 or <0.05)
#           3. flagged BLOWN regions  -> re-segment using the DARKEST bracket
#           4. flagged CRUSHED regions -> re-segment using the BRIGHTEST bracket
#           5. merge: 0EV masks (default) + overrides (flagged regions only)
# "fused" = one segmentation of the quick fusion   ·   "mid" = 0EV only
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import torch
from PIL import Image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if "of_model" not in globals():
    try:
        OF_ID = "shi-labs/oneformer_ade20k_dinat_large"
        of_proc = OneFormerProcessor.from_pretrained(OF_ID)
        of_model = OneFormerForUniversalSegmentation.from_pretrained(OF_ID).to(DEVICE).eval()
    except Exception as e:
        print(f"DiNAT unavailable ({type(e).__name__}) — Swin-L fallback")
        OF_ID = "shi-labs/oneformer_ade20k_swin_large"
        of_proc = OneFormerProcessor.from_pretrained(OF_ID)
        of_model = OneFormerForUniversalSegmentation.from_pretrained(OF_ID).to(DEVICE).eval()


def _run_seg(bgr):
    """One OneFormer semantic pass on a BGR image -> ADE20K label map."""
    _pl = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    _in = of_proc(images=_pl, task_inputs=["semantic"], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        _ou = of_model(**_in)
    return of_proc.post_process_semantic_segmentation(
        _ou, target_sizes=[_pl.size[::-1]])[0].cpu().numpy()


if SEG_SOURCE == "multi":
    _mid = aligned[N // 2]
    seg0 = _run_seg(_mid)                                  # STEP 1 — segment 0EV
    _mf = _mid.astype(np.float32) / 255
    _clip_hi = (_mf > 0.95).any(2)                         # blown in 0EV (per channel)
    _clip_lo = (_mf < 0.05).all(2)                         # crushed in 0EV
    # STEP 2 — flag every 0EV mask (per-class component) that overlaps a clipped
    # region by >= 20%: the segmenter could not SEE that area properly at 0EV
    ov_hi = np.zeros((H, Wd), bool)
    ov_lo = np.zeros((H, Wd), bool)
    for _c in np.unique(seg0):
        _ncc, _lcc = cv2.connectedComponents((seg0 == _c).astype(np.uint8))
        for _i in range(1, _ncc):
            _comp = _lcc == _i
            if _comp.sum() < 0.0005 * H * Wd:
                continue
            if float(_clip_hi[_comp].mean()) >= 0.20:
                ov_hi |= _comp                             # window/sky-type flag
            elif float(_clip_lo[_comp].mean()) >= 0.20:
                ov_lo |= _comp                             # shadow-type flag
    # a blown WINDOW usually hides INSIDE a huge wall component at 0EV (that is
    # exactly why it needs re-segmentation!) — there the component test dilutes
    # below any threshold. So the substantial clipped BLOBS themselves are
    # flagged too, slightly grown so the re-segmentation gets context.
    _k4 = np.ones((max(5, int(Wd * 0.006)) | 1,) * 2, np.uint8)
    for _clipm, _is_hi in ((_clip_hi, True), (_clip_lo, False)):
        _ncb, _lcb, _scb, _ = cv2.connectedComponentsWithStats(_clipm.astype(np.uint8))
        _blob = np.zeros((H, Wd), bool)
        for _i in range(1, _ncb):
            if _scb[_i, cv2.CC_STAT_AREA] >= 0.001 * H * Wd:
                _blob |= _lcb == _i
        _blob = cv2.dilate(_blob.astype(np.uint8), _k4).astype(bool)
        if _is_hi:
            ov_hi |= _blob
        else:
            ov_lo |= _blob
    ov_lo &= ~ov_hi                                        # blown wins if both touch
    seg = seg0.copy()
    if ov_hi.any():                                        # STEP 3 — blown areas get the
        seg_dark = _run_seg(aligned[0])                    # DARKEST bracket's labels
        seg[ov_hi] = seg_dark[ov_hi]
    if ov_lo.any():                                        # STEP 4 — crushed areas get the
        seg_bright = _run_seg(aligned[-1])                 # BRIGHTEST bracket's labels
        seg[ov_lo] = seg_bright[ov_lo]                     # STEP 5 — merge (overrides only)
    print(f"multi-bracket segmentation: {100 * ov_hi.mean():.1f}% re-segmented from the "
          f"DARKEST bracket (blown at 0EV) · {100 * ov_lo.mean():.1f}% from the "
          f"BRIGHTEST (crushed at 0EV) · rest = 0EV masks")
    # SHOW the flags: where the 0EV segmentation was NOT trusted
    _vis = cv2.cvtColor(_mid, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    _vis[ov_hi] = _vis[ov_hi] * 0.4 + np.array([1.0, 0.2, 0.2]) * 0.6
    _vis[ov_lo] = _vis[ov_lo] * 0.4 + np.array([0.2, 0.4, 1.0]) * 0.6
    plt.figure(figsize=(18, 11))
    plt.imshow(np.clip(_vis, 0, 1)); plt.axis("off")
    plt.title("STAGE 2 — MULTI-BRACKET SEGMENTATION:  RED = re-segmented from the DARKEST "
              "bracket (blown at 0EV)  ·  BLUE = from the BRIGHTEST (crushed at 0EV)")
    plt.show()
else:
    _src = mertens if SEG_SOURCE == "fused" else aligned[N // 2]
    seg = _run_seg(_src)
    print(f"segmented the {SEG_SOURCE} image")
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(_src, cv2.COLOR_BGR2RGB)); plt.axis("off")
    plt.title(f"STAGE 2 — the exact image being segmented ({SEG_SOURCE})")
    plt.show()


# ============================== CELL 5 — STAGES 3-5: SEMANTIC FUSION ===========
# ---- the correction classes (ADE20K ids -> the target-table buckets) ----------
BUCKETS = {
    "window":           (8, 14),                            # window, glass door
    "ceiling_light":    (5, 36, 82, 85, 139),               # ceiling, lamps, fan
    "wall":             (0, 1),                             # wall, building
    "floor":            (3, 13, 28),                        # floor, ground, rug
    "furniture_wood":   (7, 15, 19, 23, 30, 33),        # bed, table, chair, sofa, desk.
                                                        # CABINETS/shelves/wardrobes are
                                                        # deliberately NOT caught — they
                                                        # follow the room (a white kitchen
                                                        # cabinet must never be darkened)
    "countertop_stone": (45,),                              # countertop
}
BMASK = {nm: np.isin(seg, ids) for nm, ids in BUCKETS.items()}
print("bucket coverage:")
for _nm, _m in BMASK.items():
    print(f"  {_nm:17s} {100 * _m.mean():5.1f}%")

# ---- SHOW the semantic segmentation as a colour mask --------------------------
BCOL = {"window": (255, 255, 0), "ceiling_light": (0, 140, 255), "wall": (255, 120, 0),
        "floor": (0, 200, 0), "furniture_wood": (0, 0, 230), "countertop_stone": (200, 0, 200)}
_ov = mertens.astype(np.float32).copy()
for _nm, _m in BMASK.items():
    _a = _m.astype(np.float32)[..., None] * 0.45
    _t = np.zeros_like(_ov); _t[:] = BCOL[_nm]
    _ov = _ov * (1 - _a) + _t * _a
plt.figure(figsize=(18, 11))
plt.imshow(cv2.cvtColor(_ov.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title("STAGE 2 — SEMANTIC SEGMENTATION:  cyan window · orange ceiling/light · "
          "blue wall · green floor · red furniture/wood · magenta counter "
          "(cabinets NOT caught — they follow the room)")
plt.show()

# ---- THE PROFILE: per-class target luma bands — the tunable, external part ----
# (values from the design doc's neutral/luxury tables; standard sits between)
PROFILES = {
    "rental":   {"window": (0.75, 0.92), "ceiling_light": (0.80, 0.95),
                 "wall": (0.45, 0.65),   "floor": (0.30, 0.55),
                 "furniture_wood": (0.25, 0.45), "countertop_stone": (0.50, 0.70),
                 "global_peak": 0.50, "s_curve": 0.00, "sat_wood": 1.00},
    "standard": {"window": (0.78, 0.93), "ceiling_light": (0.82, 0.96),
                 "wall": (0.48, 0.68),   "floor": (0.32, 0.58),
                 "furniture_wood": (0.26, 0.46), "countertop_stone": (0.52, 0.72),
                 "global_peak": 0.54, "s_curve": 0.03, "sat_wood": 1.04},
    "luxury":   {"window": (0.80, 0.95), "ceiling_light": (0.85, 0.97),
                 "wall": (0.50, 0.70),   "floor": (0.35, 0.60),
                 "furniture_wood": (0.28, 0.48), "countertop_stone": (0.55, 0.75),
                 "global_peak": 0.58, "s_curve": 0.05, "sat_wood": 1.08},
}
PROFILE = "standard"   #@param ["rental", "standard", "luxury"]
T = PROFILES[PROFILE]
print(f"profile: {PROFILE}")

# ---- STAGE 3: per-region fusion targets + best-bracket scores -----------------
# Each region is fused toward a SAFE version of its own band mid (deep windows /
# blasted shadows are prevented here; stage 6 then lifts precisely to the band).
_flu = mertens.astype(np.float32).mean(2) / 255
_fusion_groups = {}
for _nm, _m in BMASK.items():
    _lo, _hi = T[_nm]
    _fusion_groups[_nm] = (_m, float(np.clip((_lo + _hi) / 2, 0.30, 0.72)))
_rest = ~np.any(list(BMASK.values()), 0)
_fusion_groups["other"] = (_rest & (_flu >= 0.22), 0.50)
_fusion_groups["shadow pockets"] = (_rest & (_flu < 0.22), 0.32)   # depth kept

# per-pixel TARGET map (feathered — no seams). The fusion weights are computed
# AGAINST this map, so every pixel natively prefers the bracket that shows it
# at its OWN semantic target — not at generic midtone grey.
TMAP = np.full((H, Wd), 0.50, np.float32)
for _nm, (_m, _t) in _fusion_groups.items():
    TMAP[_m] = _t
TMAP = cv2.GaussianBlur(TMAP, (0, 0), max(8, int(Wd * 0.008)))

# ---- STAGE 4: semantically-modulated Mertens weights --------------------------
WBASE = []
for _im, _lu in zip(IMS, LUS):
    _con = np.abs(cv2.Laplacian(_lu, cv2.CV_32F))           # contrast
    _sat = _im.std(2)                                       # saturation
    _wex = np.exp(-((_im - TMAP[..., None]) ** 2) / 0.08).prod(2)   # target-exposedness
    WBASE.append(((_con + 1e-4) * (_sat + 1e-4) * (_wex + 1e-6)).astype(np.float32))

BOOST = 2.0   #@param {type:"number"}   # how strongly the best bracket leads its region
print("best bracket per region (score = closeness to the region's target - 2x clipped):")
for _nm, (_m, _t) in _fusion_groups.items():
    if _m.mean() < 0.001:
        print(f"  {_nm:17s} (not found)"); continue
    # GLARE GUARD (shadow pockets only): a bracket that is MAJORITY-blown can
    # still score best here — its deep shadows sit right at 0.32 and are not
    # clipped IN the region — but such a frame carries veiling glare: on glass
    # (a glass-front cabinet) its "shadow" pixels are milky reflections, and
    # fusing them bakes white halos into the pane. Shadow pixels may only come
    # from a bracket that is < 35% blown overall (fallback: all, if none).
    _cand = list(range(N))
    if _nm == "shadow pockets":
        _ok_j = [_j for _j in range(N) if float(OVER[_j].mean()) < 0.35]
        if _ok_j:
            _cand = _ok_j
    _sc = [float(np.exp(-((LUS[_j][_m] - _t) ** 2) / (2 * 0.2 ** 2)).mean())
           - 2.0 * float(OVER[_j][_m].mean() + UNDER[_j][_m].mean())
           for _j in _cand]
    _bj = _cand[int(np.argmax(_sc))]
    print(f"  {_nm:17s} (target {_t:.2f}) <- bracket #{_bj}")
    _soft = cv2.GaussianBlur(_m.astype(np.float32), (0, 0), max(8, int(Wd * 0.008)))
    WBASE[_bj] = WBASE[_bj] * (1 + BOOST * _soft)
_wsum = np.sum(WBASE, 0) + 1e-8
WN = [(w / _wsum).astype(np.float32) for w in WBASE]

# ---- SHOW the semantic knowledge + the fusion decision ------------------------
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
_i0 = ax[0].imshow(TMAP, cmap="magma", vmin=0.2, vmax=0.8)
ax[0].axis("off"); ax[0].set_title("STAGE 3 — per-pixel TARGET luma (what each region SHOULD be)")
fig.colorbar(_i0, ax=ax[0], fraction=0.03)
_winmap = np.argmax(np.stack(WN), 0)
_i1 = ax[1].imshow(_winmap, cmap="viridis", vmin=0, vmax=N - 1)
ax[1].axis("off"); ax[1].set_title("STAGE 4 — WHICH BRACKET feeds each pixel (0 = darkest)")
fig.colorbar(_i1, ax=ax[1], fraction=0.03)
plt.tight_layout(); plt.show()

# ---- STAGE 5: Laplacian pyramid blend — the weights choose WHAT, the pyramid
# makes every transition seamless (no halos, no region edges) -------------------
_lev = max(4, int(np.log2(min(H, Wd))) - 5)
_acc = None
for _im, _w in zip(IMS, WN):
    _gi = [_im]; _gw = [_w]
    for _l in range(_lev):
        _gi.append(cv2.pyrDown(_gi[-1]))
        _gw.append(cv2.pyrDown(_gw[-1]))
    _lp = [_gi[_l] - cv2.pyrUp(_gi[_l + 1], dstsize=(_gi[_l].shape[1], _gi[_l].shape[0]))
           for _l in range(_lev)] + [_gi[-1]]
    if _acc is None:
        _acc = [np.zeros_like(_x) for _x in _lp]
    for _l in range(_lev + 1):
        _acc[_l] = _acc[_l] + _lp[_l] * _gw[_l][..., None]
_res = _acc[-1]
for _l in range(_lev - 1, -1, -1):
    _res = cv2.pyrUp(_res, dstsize=(_acc[_l].shape[1], _acc[_l].shape[0])) + _acc[_l]
fused = (np.clip(_res, 0, 1) * 255).astype("uint8")

fig, ax = plt.subplots(1, 2, figsize=(18, 6))
ax[0].imshow(cv2.cvtColor(mertens, cv2.COLOR_BGR2RGB)); ax[0].axis("off")
ax[0].set_title("blind Mertens fusion")
ax[1].imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)); ax[1].axis("off")
ax[1].set_title("STAGE 5 — SEMANTIC FUSION: each region from its best bracket")
plt.tight_layout(); plt.show()
show_hist(fused, "STAGE 5 — semantic fused (before correction)")


# ============================== CELL 6 — STAGE 6+7: DATA-DRIVEN CORRECTION =====
# For every bucket: MEASURE where it actually sits, compare to its PROFILE band,
# compute the gamma that moves it into the band — feathered, with a luminance
# band-guard so a dark corner inside the same mask is never crushed. Then verify
# and iterate (usually converges in 1-2 rounds). Nothing here is a fixed preset:
# the amounts always come from measured-vs-target.
res = fused.astype(np.float32) / 255
for _it in range(2):
    lu = res.mean(2)
    adjust = np.ones((H, Wd), np.float32)
    moved = 0
    print(f"\nround {_it + 1} — bucket        current   band          action")
    for _nm, _m in BMASK.items():
        _mm = _m if _nm == "window" else (_m & ~BMASK["window"])
        if _mm.mean() < 0.002:
            print(f"  {_nm:17s} (not found)"); continue
        _cur = float(np.median(lu[_mm]))
        _lo, _hi = T[_nm]
        _tgt = float(np.clip(_cur, _lo, _hi))
        if abs(_tgt - _cur) < 0.015 or _cur < 1e-3:
            print(f"  {_nm:17s}  {_cur:.2f}    {_lo:.2f}-{_hi:.2f}   ok")
            continue
        _g = float(np.clip(np.log(_tgt) / np.log(_cur), 0.5, 2.4))
        _band = np.clip(1 - np.abs(lu - _cur) / 0.35, 0, 1)
        _soft = cv2.GaussianBlur(_mm.astype(np.float32), (0, 0),
                                 max(4, int(Wd * 0.01))) * _band
        adjust = adjust * (1 - _soft) + _g * _soft
        moved += 1
        print(f"  {_nm:17s}  {_cur:.2f}    {_lo:.2f}-{_hi:.2f}   -> {_tgt:.2f}  (gamma {_g:.2f})")
    res = np.clip(res, 0, 1) ** adjust[..., None]
    if moved == 0:
        break

# global peak: measured against the profile's histogram target. The window and
# ceiling/light buckets are shielded BY THEIR MASKS (not by luminance — a window
# sitting at 0.78 is below any brightness gate but must not be dragged): the
# bucket rounds above already placed them in their bands, and the global step
# must never undo that.
_lu = res.mean(2)
_bright = cv2.GaussianBlur(np.clip((_lu - 0.80) / 0.12, 0, 1), (0, 0), 15)
_keep = np.clip(_bright + cv2.GaussianBlur(
    (BMASK["window"] | BMASK["ceiling_light"]).astype(np.float32), (0, 0), 10), 0, 1)
_med = float(np.median(_lu[_keep < 0.5]))
if abs(_med - T["global_peak"]) > 0.02 and _med > 1e-3:
    _g = float(np.clip(np.log(T["global_peak"]) / np.log(_med), 0.6, 1.4))
    res = np.clip(res, 0, 1) ** ((_g - 1) * (1 - _keep[..., None]) + 1)
    print(f"\nglobal peak: {_med:.2f} -> {T['global_peak']:.2f} "
          f"(gamma {_g:.2f}, window + ceiling/lights shielded)")

# profile finish: gentle S-curve + warm-wood saturation (luxury cues)
if T["s_curve"] > 0:
    res = np.clip(res + T["s_curve"] * np.sin(2 * np.pi * (res - 0.5)) * 0.5, 0, 1)
    print(f"gentle S-curve +{T['s_curve']}")
if T["sat_wood"] > 1.0:
    _lab = cv2.cvtColor((res * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    _a = _lab[..., 1] - 128; _b = _lab[..., 2] - 128
    _warm = np.clip((_b - 8) / 10, 0, 1) * np.clip((_a + 10) / 10, 0, 1)
    _f = 1 + (T["sat_wood"] - 1) * _warm
    _lab[..., 1] = _a * _f + 128
    _lab[..., 2] = _b * _f + 128
    res = cv2.cvtColor(np.clip(_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255
    print(f"warm-wood saturation x{T['sat_wood']}")

# soft highlight rolloff — only the very top is compressed so nothing clips
_l = res.mean(2)
_scale = np.where(_l > 0.93, (0.93 + (_l - 0.93) * 0.5) / np.maximum(_l, 1e-6), 1.0)
final = (np.clip(res * _scale[..., None], 0, 1) * 255).astype("uint8")


# ============================== CELL 7 — VERIFY + OUTPUT =======================
show_hist(final, "STAGE 6 — after data-driven correction")
lu = final.astype(np.float32).mean(2) / 255
print("\nVERIFICATION — per-bucket final position vs the profile band:")
for _nm, _m in BMASK.items():
    if _m.mean() < 0.002:
        continue
    _cur = float(np.median(lu[_m]))
    _lo, _hi = T[_nm]
    _okc = "OK " if (_lo - 0.03) <= _cur <= (_hi + 0.03) else "OFF"
    print(f"  [{_okc}] {_nm:17s} {_cur:.2f}   band {_lo:.2f}-{_hi:.2f}")
print(f"  clipping: blown>0.95 = {100 * (lu > 0.95).mean():.1f}%   "
      f"crushed<0.05 = {100 * (lu < 0.05).mean():.1f}%   median = {np.median(lu):.2f}")

cv2.imwrite("semantic_result.jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 95])
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
ax[0].imshow(cv2.cvtColor(mertens, cv2.COLOR_BGR2RGB)); ax[0].axis("off")
ax[0].set_title("blind Mertens fusion")
ax[1].imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB)); ax[1].axis("off")
ax[1].set_title(f"SEMANTIC PIPELINE result — profile '{PROFILE}'")
plt.tight_layout(); plt.show()
plt.figure(figsize=(18, 11))
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title(f"FINAL — semantic fusion + data-driven correction ({PROFILE})")
plt.show()
try:
    from google.colab import files
    files.download("semantic_result.jpg")
except Exception:
    pass
print("done — semantic_result.jpg")


# ============================== CELL 8 — BRIDGE to the AutoHDR chain ===========
# For the FULL AutoHDR-level look (white walls, white bulbs, cast cleaning,
# floor tone — the whole calibrated chain), run THIS cell after CELL 5, then run
# CELL 9 -> CELL 10 -> CELL 12 below (they are the proven AutoHDR chain, copied
# into this script). They read `fused` — which is now the SEMANTIC fusion — so
# the calibrated chain runs on the better base image. This bridge prepares the
# two things they need (PARAMS + files).
# FINAL result = CELL 12's download: hdr_result_local.jpg (or CELL 11's
# hdr_result_walls.jpg — the per-wall colour approach)
try:
    from google.colab import files
except Exception:
    class _F:
        def download(self, *a, **k): pass
    files = _F()


def derive_params(mertens_u8):
    """Histogram-driven per-scene tone levers (verbatim from the old CELL 2)."""
    lu = mertens_u8.astype(np.float32).mean(2) / 255
    p50 = float(np.median(lu))
    crushed = float((lu < 0.08).mean()); blown = float((lu > 0.94).mean())
    contrast = float(lu.std())
    P = {
        "shadow_fill": float(np.clip(0.18 + crushed * 2.2, 0.15, 0.45)),
        "recover":     float(np.clip(0.40 + blown * 3.5,   0.40, 0.90)),
        "expose_tgt":  float(np.clip(0.70 + (0.52 - p50) * 0.30, 0.64, 0.76)),
        "scurve":      float(np.clip(0.085 - contrast * 0.16, 0.030, 0.075)),
    }
    return P, dict(median=round(p50, 2), crushed_pct=round(crushed * 100, 1),
                   blown_pct=round(blown * 100, 1), contrast=round(contrast, 3))


PARAMS, _hist8 = derive_params(fused)
if "EVS" not in globals():
    EVS = [None] * len(aligned)        # bracket EV labels (EXIF) — display only here
print("PARAMS measured on the SEMANTIC fused image:", {k: round(v, 3) for k, v in PARAMS.items()})
print("READY — now run CELL 9 -> CELL 10, then CELL 11 (per-wall colour HDR) "
      "and/or CELL 12 (the full fused-based chain) below.")
print("CELL 11 result = hdr_result_walls.jpg · CELL 12 result = hdr_result_local.jpg")


# ============================== CELL 9 — EXPOSURE ANALYSIS =====================
# Two views that tell you WHERE the image needs more / less exposure — the same
# signal a multi-exposure model computes internally, made visible.
H, Wd = fused.shape[:2]
luma = fused.astype(np.float32).mean(2) / 255

# 1) ZONE heat-map — where the FUSED image needs more / less exposure.
# NOTE this is measured on the FUSED image, which is deliberately DARK (median
# ~0.54) before the finishing chain lifts it to ~0.73 — so a lot of "+EV" here is
# EXPECTED and is exactly what CELL 4 then applies.
# The GREEN band is deliberately NARROW (0.45-0.75 = actually well exposed). An
# earlier version used 0.30-0.78, which wrongly painted genuinely dark areas green.
ZONES = [
    (0.00, 0.10, (255,  60,   0), "crushed  (big +EV)"),
    (0.10, 0.28, (255, 200,   0), "too dark (+EV)"),
    (0.28, 0.45, (200, 255,   0), "a bit dark (small +EV)"),
    (0.45, 0.75, (0,   200,   0), "well exposed (ok)"),
    (0.75, 0.92, (0,   200, 255), "bright (ok)"),
    (0.92, 1.01, (0,     0, 255), "blown (-EV)"),
]
zone = np.zeros((H, Wd, 3), np.uint8)
print("exposure zones (of the FUSED image, before CELL 4 brightens it):")
for _lo, _hi, _col, _lbl in ZONES:
    _mk = (luma >= _lo) & (luma < _hi)
    zone[_mk] = _col
    print(f"   {_lbl:24s} {100*_mk.mean():5.1f}%")
zone_ov = (fused * 0.45 + zone * 0.55).clip(0, 255).astype("uint8")


# ---- turn the analysis into a DODGE MAP that CELL 4 actually APPLIES ----------
# The zone analysis is not just a picture: it becomes a smooth per-pixel exposure
# map. Dark zones get a local lift (+EV where the map said "+EV"), the blown end
# is already handled by highlight-recover/sun-tone, and TRUE BLACKS are anchored:
# the deepest shadows (a dark lamp body, the shadow core under furniture) are
# excluded from the lift so real blacks STAY black and the image keeps its depth.
# The map is blurred very wide -> it is a smooth dodge, incapable of halos.
def _analysis_dodge_map(fused_u8, strength=0.55):
    lu = fused_u8.astype(np.float32).mean(2) / 255
    w_dark = np.clip((0.42 - lu) / 0.42, 0, 1) ** 1.2      # how far below "ok" each px sits
    true_black = np.clip((0.06 - lu) / 0.06, 0, 1)          # deepest blacks: keep black
    w_dark = w_dark * (1 - true_black)                      # never lift a real black
    w_dark = cv2.GaussianBlur(w_dark, (0, 0), max(20, int(fused_u8.shape[1] * 0.02)))
    return 1.0 / (1.0 + strength * w_dark)                  # per-pixel gamma (<1 = brighten)


EXPO_MAP = _analysis_dodge_map(fused)
print(f"dodge map built from the analysis: lifts {100*float((EXPO_MAP<0.97).mean()):.0f}% of the image, "
      f"true blacks anchored (never lifted)")

# 2) BEST-BRACKET map: which EV each area is best-exposed in (blue=darker .. red=brighter)
wexp = [np.exp(-((im.astype(np.float32) / 255 - 0.5) ** 2) / (2 * 0.2 ** 2)).prod(2) for im in aligned]
best = np.argmax(np.stack(wexp, 0), 0)
N = len(aligned)
cmap = cv2.applyColorMap((best * 255 // max(N - 1, 1)).astype(np.uint8), cv2.COLORMAP_JET)
best_ov = (fused * 0.35 + cmap * 0.65).clip(0, 255).astype("uint8")

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
ax[0].imshow(cv2.cvtColor(zone_ov, cv2.COLOR_BGR2RGB)); ax[0].axis("off")
ax[0].set_title("EXPOSURE ZONES (of FUSED) — blue/cyan=+EV · yellow-green=slightly +EV · green=ok · red=-EV")
ax[1].imshow(cv2.cvtColor(best_ov, cv2.COLOR_BGR2RGB)); ax[1].axis("off")
ax[1].set_title("BEST BRACKET per area — blue=darker EV · red=brighter EV")
plt.tight_layout(); plt.show()

print(f"blown>0.95 = {100*(luma>0.95).mean():.1f}%   crushed<0.12 = {100*(luma<0.12).mean():.1f}%")
if "EVS" not in globals():
    EVS = [None] * N                   # EV labels (EXIF) — optional, "?" shown without them
for i in range(N):
    ev = f"EV{EVS[i]:+g}" if EVS[i] is not None else "?"
    print(f"  bracket #{i} ({ev:>6s}) is best-exposed for {100*(best==i).mean():5.1f}% of the image")


# ============================== CELL 10 — the calibrated FINISHING chain ========
# Produces the AutoHDR-level look. Every step is calibrated (measured against an
# AutoHDR reference); the comments say what each one guarantees.
def _wb(img, pct=92, exclude=None):                    # white-patch white balance
    lu = img.mean(2); valid = lu < 0.98
    if exclude is not None: valid &= exclude < 0.5
    v = lu[valid]
    if v.size < 1000: return img
    ref = img[valid & (lu >= np.percentile(v, pct))]
    if len(ref) < 100: return img
    g = ref.mean(0); g = g.max() / np.maximum(g, 1e-6)
    return np.clip(img * g, 0, 1)


def _levels_soft(img, blk=0.4, wht=99.5, maxblk=0.25, knee=0.82, exclude=None):
    """Black/white-point stretch with a SOFT SHOULDER — near-white values are
    compressed gently (tanh) instead of hard-clipped, so a white sink / counter
    KEEPS its detail (highlights maintained) while walls still read clean white."""
    lu = img.mean(2); s2 = lu if exclude is None else lu[exclude < 0.5]
    if s2.size < 1000: s2 = lu
    lo = min(float(np.percentile(s2, blk)), maxblk); hi = float(np.percentile(s2, wht))
    if hi - lo < 0.1: return img
    x = (img - lo) / (hi - lo)
    x = np.where(x <= knee, np.clip(x, 0, None),
                 knee + (1 - knee) * np.tanh((x - knee) / (1 - knee)))
    return np.clip(x, 0, 1) * (0.93 - 0.02) + 0.02


def _bright_ramp(img):                                 # soft mask of blown areas (window/lights)
    lu = img.mean(2)
    ramp = np.clip((lu - 0.80) / 0.12, 0, 1)
    k = max(31, int(img.shape[1] * 0.008) | 1)
    return cv2.GaussianBlur(ramp, (k, k), 0)


def _expose(img, target=0.68, exclude=None):
    """Gamma-lift the room to `target`, measured on NON-bright pixels only and
    FADED OUT over the bright mask — brightens the interior WITHOUT washing out
    the window/lights (this is the anti-wash-out step)."""
    lu = img.mean(2); sel = lu if exclude is None else lu[exclude < 0.5]
    if sel.size < 1000: sel = lu
    m = max(float(np.median(sel)), 1e-3)
    if m >= target: return img
    gamma = float(np.clip(np.log(target) / np.log(m), 0.45, 1.0))
    lifted = np.clip(img, 0, 1) ** gamma
    if exclude is None: return lifted
    ex = exclude[..., None]
    return lifted * (1 - ex) + np.clip(img, 0, 1) * ex


def _neutralize(img):
    """De-tint surfaces that should be neutral -> SMOOTH WALL COLOURS + natural
    light. Four paths: bright whites, shadow de-tint (kills the dirty warm cast
    on dark surfaces), a halo-band term (the tan strip where a whitened ceiling
    meets a wall) and a bulb-glow term (a warm lamp's glow on a white ceiling).
    a-gates hard-protect green walls and pink/wood so real colour never moves."""
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]/255; a = lab[..., 1]-128; b = lab[..., 2]-128
    ch = np.sqrt(a*a+b*b)
    wgt = np.clip((L-0.70)/0.12, 0, 1) * np.clip((22-ch)/8, 0, 1)
    wgt = np.maximum(wgt, np.clip((L-0.85)/0.10, 0, 1))
    shadow_wgt = np.clip((14-ch)/6, 0, 1) * np.clip((L-0.04)/0.05, 0, 1) * 0.5
    wgt = np.maximum(wgt, shadow_wgt)
    ga = np.clip((a+5)/3, 0, 1) * np.clip((4-a)/4, 0, 1)
    strength = 0.55 + 0.35*np.clip((0.5-a)/2, 0, 1)
    yellow_wgt = (ga * np.clip((b-4)/6, 0, 1) * np.clip((28-ch)/8, 0, 1)
                  * np.clip((L-0.45)/0.12, 0, 1) * strength)
    sc = ch - 2*a
    glow_wgt = (np.clip((6-a)/4, 0, 1) * np.clip((13.5-sc)/3.5, 0, 1)
                * np.clip((b-4)/6, 0, 1) * np.clip((L-0.45)/0.12, 0, 1) * 0.9)
    yellow_wgt = np.maximum(yellow_wgt, glow_wgt)
    wgt = cv2.GaussianBlur(wgt, (0, 0), 8)
    wgt = np.maximum(wgt, cv2.GaussianBlur(yellow_wgt, (0, 0), 3))
    lab[..., 1] = a*(1-wgt)+128; lab[..., 2] = b*(1-wgt)+128
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _match_white(img, strength=0.85):                  # push the brightest pixels to exact neutral
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    br = lab[..., 0] > 0.78*255
    if br.sum() < 1000: return img
    lab[..., 1] += (128-float(lab[..., 1][br].mean()))*strength
    lab[..., 2] += (128-float(lab[..., 2][br].mean()))*strength
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _lift_whites(img, amount=0.16, start=0.45, exclude=None):
    """Brighten only the bright zone toward white; the (1-img) term self-limits
    near white (keeps highlight detail) and `exclude` keeps the window out."""
    lu = img.mean(2)
    w = np.clip((lu - start) / (1.0 - start), 0, 1) ** 1.3
    if exclude is not None: w = w * (1 - exclude)
    return np.clip(img + amount * w[..., None] * (1.0 - img), 0, 1)


def _scurve(img, s2=0.045):                            # gentle contrast (kept low so highlights hold)
    return np.clip(img + s2*np.sin(2*np.pi*(img-0.5)), 0, 1)


def _tame_warm(img, knee=16, compress=0.55, l_gain=24):
    """Browns/oranges (wood, terracotta): soft-knee the chroma + give dark warm
    pixels a little L back — stops them going over-saturated / too dark."""
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    A = lab[..., 1]-128; B = lab[..., 2]-128
    ch = np.sqrt(A*A+B*B)
    warm_w = np.clip(A/6, 0, 1) * np.clip(B/10, 0, 1)
    over = np.clip(ch - knee, 0, None)
    scale = (knee + over*compress) / np.maximum(ch, 1e-3)
    scale = 1 + (scale - 1) * warm_w
    scale = cv2.GaussianBlur(scale.astype(np.float32), (0, 0), 3)
    lab[..., 1] = A*scale + 128; lab[..., 2] = B*scale + 128
    lw = np.clip((ch-8)/20, 0, 1) * warm_w * np.clip((0.62 - lab[..., 0]/255)/0.35, 0, 1)
    lw = cv2.GaussianBlur(lw.astype(np.float32), (0, 0), 3)
    lab[..., 0] = np.clip(lab[..., 0] + l_gain*lw, 0, 255)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _desat_warm(img, lo=30, hi=115):
    """Desaturate yellow/orange CASTS but PROTECT strong real colour: weak
    chroma (<=14 — light casts) fully desaturated, strong chroma (>=26 — real
    wood/decor) keeps 75% of its saturation. Greens hard-protected by the a-gate."""
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1]-128; b = lab[..., 2]-128
    ch = np.sqrt(a*a + b*b)
    hue = np.degrees(np.arctan2(b, a))
    w = np.clip((hue-lo)/18, 0, 1) * np.clip((hi-hue)/12, 0, 1)
    w *= np.clip(b/6, 0, 1) * np.clip((a+6)/4, 0, 1)
    keep = np.clip((ch - 14) / 12, 0, 1)
    w *= 1 - 0.75*keep
    w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), 3)
    amount = 0.4 + 0.2*np.clip((hue-70)/15, 0, 1)
    f = 1 - amount*w
    lab[..., 1] = a*f + 128; lab[..., 2] = b*f + 128
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _pinch_warm(img, amount=2.2):
    """A PINCH of warmth in the MID-TONES only (not the neutralised whites, not
    the shadows) so the image feels natural/inviting instead of clinical — never
    enough to read as a cast."""
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]/255
    w = np.clip((L-0.25)/0.25, 0, 1) * np.clip((0.90-L)/0.20, 0, 1)
    w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), 6)
    lab[..., 2] = np.clip(lab[..., 2] + amount*w, 0, 255)          # +b = warmer
    lab[..., 1] = np.clip(lab[..., 1] + 0.4*amount*w, 0, 255)      # a touch of +a
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _sharpen(img, fine=0.9, clarity=0.2):
    """Fine unsharp + wide clarity, with clarity DAMPED in shadows (full clarity
    on dark pixels is what reads as muddy/gritty) — no halos."""
    img = np.clip(img + fine*(img - cv2.GaussianBlur(img, (0, 0), 1.0)), 0, 1)
    wide = cv2.GaussianBlur(img, (0, 0), 15)
    lu = img.mean(2)
    damp = np.clip((lu - 0.25) / 0.25, 0.15, 1.0)[..., None]
    return np.clip(img + clarity*damp*(img - wide), 0, 1)


def _soft_highlights(img, knee=0.90, slope=0.55):
    """Soft-compress ONLY the near-blown highlights (luma > knee) so bright
    walls / windows / speculars keep tone instead of clipping to paper white.
    Measured: AutoHDR sits at ~0.7% blown while ours was ~5% — this closes most
    of that gap. Monotonic ratio on luminance -> cannot halo or ring."""
    lu = img.mean(2)
    if not (lu > knee).any():
        return img
    new = np.where(lu > knee, knee + (lu - knee) * slope, lu)
    ratio = np.where(lu > 1e-6, new / np.maximum(lu, 1e-6), 1.0)
    return np.clip(img * ratio[..., None], 0, 1)


def _auto_warm_amount(img, target_b=2.0):
    """ADAPTIVE warmth: measure the scene's own mid-tone warmth AFTER the cast
    cleanup, and return only the pinch needed to land at a natural target
    (LAB b ~ +2). An already-warm scene gets ~0 (warmth is never stacked on
    warmth); a clinical/cold one gets a real pinch. ~0.30 b per pinch unit
    (measured), capped so it can never look like a cast."""
    lab = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] / 255
    mid = (L > 0.30) & (L < 0.85)
    if mid.sum() < 1000:
        return 0.0
    cur_b = float((lab[..., 2] - 128)[mid].mean())
    return float(np.clip((target_b - cur_b) / 0.30, 0.0, 3.5))


def _auto_lift_amount(img, base=0.16):
    """ADAPTIVE white-lift: scale the walls/whites lift by the image's blowing
    HEADROOM — a scene already near clipping gets a smaller lift (protects
    highlights), a dim scene gets the full push."""
    blown = float((img.mean(2) > 0.94).mean())
    return base * float(np.clip((0.03 - blown) / 0.03, 0.25, 1.0))


# ---- run the chain ----
AUTO_WARMTH = True   #@param {type:"boolean"}  # measure the scene, add only the warmth it lacks
PINCH_WARMTH = 2.2   #@param {type:"number"}   # used only when AUTO_WARMTH is off

img = fused.astype(np.float32) / 255
ex = _bright_ramp(img)                                 # shield the window/lights from the lifts
img = _wb(img, exclude=ex)
img = _levels_soft(img, exclude=ex)                    # highlights maintained
img = _expose(img, target=PARAMS["expose_tgt"], exclude=ex)   # dynamic — no wash-out
if "EXPO_MAP" in globals():                            # CELL 3's analysis DRIVES the exposure:
    img = np.clip(img, 0, 1) ** EXPO_MAP[..., None]    # dark zones lifted, true blacks kept
img = _neutralize(img)                                 # smooth walls / natural light
img = _match_white(img)
_lift = _auto_lift_amount(img)                         # dynamic — less lift near clipping
img = _lift_whites(img, amount=_lift, exclude=ex)
img = _scurve(img, PARAMS["scurve"])                   # dynamic contrast (flat scenes get more)
img = _tame_warm(img)
img = _desat_warm(img)
_pw = _auto_warm_amount(img) if AUTO_WARMTH else PINCH_WARMTH
img = _pinch_warm(img, _pw)                            # dynamic — only the warmth the scene lacks
img = _sharpen(img)
img = _soft_highlights(img)                            # keep highlight tone (no paper-white clip)
print(f"adaptive: lift={_lift:.2f}  warmth pinch={_pw:.1f} ({'auto' if AUTO_WARMTH else 'manual'})")
result = (img * 255).clip(0, 255).astype("uint8")
cv2.imwrite("hdr_result.jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])

# BEFORE / AFTER histograms
show_hist(fused, "FUSED (before)")
show_hist(result, "AUTOHDR-LEVEL RESULT (after)")

# the RESULT on its own, full width (so you can see it individually)
plt.figure(figsize=(18, 11))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title("HDR RESULT (CELL 4)")
plt.show()

files.download("hdr_result.jpg")
print("done — hdr_result.jpg")


# ============================== CELL 11 — PER-WALL COLOUR HDR (walls-first) ====
# NEW APPROACH: decide each wall's TRUE colour FIRST, then apply the HDR per wall.
#   * WHITE walls    -> the AutoHDR white look: bright, clean, even (L ~0.82)
#   * COLOURED walls -> the SAME light treatment, but the wall KEEPS ITS OWN
#     COLOUR — enhanced as that colour, never whitened, never bleached
# Segmentation + Gemini are used ONLY to find the walls and set each one's
# colour — nothing else; every pixel edit is deterministic math. Standalone
# output: hdr_result_walls.jpg — compare it directly against CELL 12's result.
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if "of_model" not in globals():
    try:
        OF_ID = "shi-labs/oneformer_ade20k_dinat_large"
        of_proc = OneFormerProcessor.from_pretrained(OF_ID)
        of_model = OneFormerForUniversalSegmentation.from_pretrained(OF_ID).to(DEVICE).eval()
    except Exception as e:
        print(f"DiNAT unavailable ({type(e).__name__}) — Swin-L fallback")
        OF_ID = "shi-labs/oneformer_ade20k_swin_large"
        of_proc = OneFormerProcessor.from_pretrained(OF_ID)
        of_model = OneFormerForUniversalSegmentation.from_pretrained(OF_ID).to(DEVICE).eval()

from PIL import Image
pil = Image.fromarray(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
inp = of_proc(images=pil, task_inputs=["semantic"], return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = of_model(**inp)
wseg = of_proc.post_process_semantic_segmentation(out, target_sizes=[pil.size[::-1]])[0].cpu().numpy()

W_WINDOW = np.isin(wseg, (8, 14))
w_wallm = np.isin(wseg, (0, 1)) & ~W_WINDOW        # the walls (ceiling handled below)
w_ceil = (wseg == 5) & ~W_WINDOW

# ---- the cast-corrected COLOUR SOURCE (global WB only — paint colour survives,
# the tungsten tint goes; same role as CELL 12's fusedC) -------------------------
w_img = fused.astype(np.float32) / 255
w_ex = _bright_ramp(w_img)
w_img = _wb(w_img, exclude=w_ex)
w_fC = (np.clip(w_img, 0, 1) * 255).astype("uint8")
w_lab = cv2.cvtColor(w_fC, cv2.COLOR_BGR2LAB).astype(np.float32)
w_La = w_lab[..., 0] / 255
w_a = w_lab[..., 1] - 128; w_b = w_lab[..., 2] - 128

# ---- STEP 1: how many wall COLOURS does the room have? (deterministic 2-means)
# SEAM SAFETY: a single white wall under a warm-light GRADIENT is a colour
# CONTINUUM — its k-means halves sit close together and FAIL the separation test
# (>= 9 ab-units apart AND each side >= 8% of the wall), so it stays ONE colour:
# no split, no seam. Only a REAL second paint (an accent wall, coloured tile
# wainscot) forms a separated second mode.
w_two = False
c1 = c2 = np.zeros(2, np.float32)
if w_wallm.sum() > 4000:
    w_pts = np.stack([w_a[w_wallm][::4], w_b[w_wallm][::4]], 1).astype(np.float32)
    _chp = np.sqrt((w_pts ** 2).sum(1))
    _o = np.argsort(_chp)
    c1 = np.median(w_pts[_o[:max(1, len(_o) // 4)]], 0)      # low-chroma quarter start
    c2 = np.median(w_pts[_o[-max(1, len(_o) // 4):]], 0)     # high-chroma quarter start
    _as2 = np.zeros(len(w_pts), bool)
    for _it in range(12):
        _d1 = ((w_pts - c1) ** 2).sum(1); _d2 = ((w_pts - c2) ** 2).sum(1)
        _as2 = _d2 < _d1
        if _as2.sum() < 50 or (~_as2).sum() < 50:
            break
        c1 = w_pts[~_as2].mean(0); c2 = w_pts[_as2].mean(0)
    _sep = float(np.hypot(c2[0] - c1[0], c2[1] - c1[1]))
    _share = float(_as2.mean())
    w_two = bool(_sep >= 9.0 and 0.08 <= _share <= 0.92)
    print(f"wall colour modes: separation {_sep:.1f} ab-units, second-mode share "
          f"{100*_share:.0f}% -> {'TWO wall colours' if w_two else 'ONE wall colour'}")

w_groups = []                                       # [name, mask]
if w_two:
    _d1f = (w_a - c1[0]) ** 2 + (w_b - c1[1]) ** 2
    _d2f = (w_a - c2[0]) ** 2 + (w_b - c2[1]) ** 2
    w_cl2 = (_d2f < _d1f) & w_wallm
    # spatial cleanup: the split may not speckle — smooth it, drop tiny islands
    w_cl2 = (cv2.GaussianBlur(w_cl2.astype(np.float32), (0, 0), 8) > 0.5) & w_wallm
    _ni, _li, _si, _ = cv2.connectedComponentsWithStats(w_cl2.astype(np.uint8))
    for _i2 in range(1, _ni):
        if _si[_i2, cv2.CC_STAT_AREA] < 0.003 * H * Wd:
            w_cl2[_li == _i2] = False
    if w_cl2.any() and (w_wallm & ~w_cl2).any():
        w_groups = [["wall A", w_wallm & ~w_cl2], ["wall B", w_cl2]]
    else:
        w_two = False
if not w_two:
    w_groups = [["walls", w_wallm.copy()]]

# measured colour per group (on the cast-corrected fused — the paint's own tone)
for _g in w_groups:
    _gm = _g[1]
    if _gm.sum() > 500:
        _g += [float(np.median(w_a[_gm])), float(np.median(w_b[_gm])),
               float(np.median(np.sqrt(w_a[_gm] ** 2 + w_b[_gm] ** 2)))]
    else:
        _g += [0.0, 0.0, 0.0]

# ---- STEP 2: each group's VERDICT — white or coloured --------------------------
# Measurement guards decide the clear cases outright; Gemini judges the middle
# band (a beige paint vs a warm-lit white wall) exactly like the object judge:
# one small JSON, zero pixel edits. No key / no network -> chroma heuristic.
# tile-texture share per group (tiles/marble keep their natural tone — the
# long-standing user rule): a warm wall group that is mostly TILED (a wainscot,
# a stone wall) is a material, not cast — it defaults to coloured.
_wtex = cv2.GaussianBlur(np.abs(w_La - cv2.GaussianBlur(w_La, (0, 0), 3)),
                         (0, 0), max(10, int(Wd * 0.015)))
_wtile_share = [float((_wtex[_g[1]] > 0.026).mean()) if _g[1].sum() > 500 else 0.0
                for _g in w_groups]
for _g, _tsh in zip(w_groups, _wtile_share):
    # heuristic default — the mud-vs-colour axis: warm-light CAST on white paint
    # is pure YELLOW (b positive, a near 0), while real paint colour carries its
    # own hue (green a<0, blue b<0, beige a>=+5). Near-neutral always reads
    # white; pure-yellow low-a reads white UNLESS the group is mostly tiled
    # (then its warmth is the tile's own tone and must be kept).
    _mid_white = (_g[2] < 5.0 and _g[3] > 0.0 and _g[4] < 16.0 and _tsh < 0.30)
    _g.append("white" if (_g[4] < 9.0 or _mid_white) else "coloured")
    _g.append("")                                            # tone name (filled by LLM)
W_USE_LLM = True   #@param {type:"boolean"}
if W_USE_LLM and any(4.5 <= _g[4] <= 18.0 for _g in w_groups):
    try:
        import os as _os, base64 as _b64, json as _js, urllib.request as _ur
        _wkey = _os.environ.get("GEMINI_API_KEY", "")
        if not _wkey:
            try:
                from google.colab import userdata as _wud
                _wkey = _wud.get("GEMINI_API_KEY") or ""
            except Exception:
                pass
        if not _wkey and _os.path.exists(".env"):
            for _ln in open(".env"):
                if _ln.startswith("GEMINI_API_KEY="):
                    _wkey = _ln.split("=", 1)[1].strip()
        _wkey = "".join(_wkey.split())
        if not _wkey:
            raise RuntimeError("no GEMINI_API_KEY")
        _wann = fused.copy()
        for _wi, _g in enumerate(w_groups, 1):
            _nn, _ll, _ss, _ = cv2.connectedComponentsWithStats(_g[1].astype(np.uint8))
            if _nn < 2:
                continue
            _big = 1 + int(np.argmax(_ss[1:, cv2.CC_STAT_AREA]))
            _ys, _xs = np.where(_ll == _big)
            cv2.rectangle(_wann, (int(_xs.min()), int(_ys.min())),
                          (int(_xs.max()), int(_ys.max())), (0, 0, 255), max(2, Wd // 500))
            cv2.putText(_wann, str(_wi), (int(_xs.min()) + 8, int(_ys.min()) + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        _wann = cv2.resize(_wann, (768, int(_wann.shape[0] * 768 / _wann.shape[1])))
        _wjpg = cv2.imencode(".jpg", _wann, [cv2.IMWRITE_JPEG_QUALITY, 88])[1]
        _wprompt = (
            'You are a real-estate photo analyst. This is the UNEDITED fused photo of a '
            'room under warm/yellow light. Red numbered boxes mark WALL areas. For each, '
            'judge the TRUE PAINT colour, ignoring the colour of the light falling on it.\n'
            'Return ONLY this JSON:\n'
            '{"walls": [{"id": <box number>, "paint": "white" or "coloured", '
            '"tone": "<one/two words, e.g. sage green, beige, light grey — empty if white>"}]}\n'
            'Rules: "white" = white/off-white paint (it may LOOK yellow under warm bulbs '
            'or shaded grey). "coloured" = genuinely painted a colour or faced with '
            'coloured tile. Judge every numbered box.')
        _wbody = {"contents": [{"parts": [
                     {"inline_data": {"mime_type": "image/jpeg",
                                      "data": _b64.b64encode(_wjpg.tobytes()).decode()}},
                     {"text": _wprompt}]}],
                  "generationConfig": {"temperature": 0, "response_mime_type": "application/json"}}
        _wresp = None
        _wlast = ""
        for _wm in ("gemini-pro-latest", "gemini-flash-latest", "gemini-2.0-flash"):
            try:
                _wrq = _ur.Request(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{_wm}:generateContent?key={_wkey}",
                    data=_js.dumps(_wbody).encode(), headers={"Content-Type": "application/json"})
                with _ur.urlopen(_wrq, timeout=90) as _wr:
                    _wresp = _js.loads(_wr.read())
                break
            except Exception as _we:
                _wlast = f"{_wm}: {type(_we).__name__}"
                continue
        if _wresp is None:
            raise RuntimeError(_wlast.replace(_wkey, "[KEY]"))
        _wtxt = "".join(_p["text"] for _p in _wresp["candidates"][0]["content"]["parts"]
                        if isinstance(_p.get("text"), str))
        print("wall judge — Gemini's raw answer:", _wtxt.strip()[:400])
        _wdec = _js.JSONDecoder()
        _wobj = None
        _wpos = _wtxt.find("{")
        while _wpos != -1 and _wobj is None:
            try:
                _cand = _wdec.raw_decode(_wtxt[_wpos:])[0]
                if isinstance(_cand, dict) and "walls" in _cand:
                    _wobj = _cand
            except Exception:
                pass
            _wpos = _wtxt.find("{", _wpos + 1)
        if _wobj is None and "{" in _wtxt:
            # the answer itself was TRUNCATED (missing closing braces) — the same
            # failure the object judge already handles: complete the braces
            _wfix = _wtxt[_wtxt.find("{"):]
            _wfix = _wfix + "]" * max(0, _wfix.count("[") - _wfix.count("]"))
            _wfix = _wfix + "}" * max(0, _wfix.count("{") - _wfix.count("}"))
            try:
                _cand = _wdec.raw_decode(_wfix)[0]
                if isinstance(_cand, dict) and "walls" in _cand:
                    _wobj = _cand
                    print("(truncated wall answer repaired by completing the braces)")
            except Exception:
                pass
        if _wobj is None:
            raise RuntimeError("no valid JSON in the answer")
        for _o2 in _wobj.get("walls", []):
            _wi = int(_o2.get("id", 0)) - 1
            if 0 <= _wi < len(w_groups):
                w_groups[_wi][5] = str(_o2.get("paint", w_groups[_wi][5]))
                w_groups[_wi][6] = str(_o2.get("tone", ""))
    except Exception as _we2:
        print(f"wall judge unavailable ({type(_we2).__name__}: {str(_we2)[:150]}) — "
              f"using the chroma heuristic")
# TILE GUARD (outranks the LLM): a wall group that is mostly TILED with its own
# warm tone is a MATERIAL — the judge was asked about paint, but tiles keep
# their natural tone whatever the verdict (the long-standing tile rule; AutoHDR
# keeps the bathroom wainscot at b+7..+12). EXCEPT in a genuinely WHITE scene
# (almost no real colour anywhere): there the tile's warmth is lighting cast on
# white tile and AutoHDR neutralises it (measured: white-kitchen backsplash b+0).
_wvalid = (w_La > 0.15) & (w_La < 0.95) & ~W_WINDOW
_wcolorful = (float((np.sqrt(w_a ** 2 + w_b ** 2)[_wvalid] > 25).mean()) * 100
              if _wvalid.sum() > 1000 else 0.0)
for _g, _tsh in zip(w_groups, _wtile_share):
    if _tsh >= 0.30 and _g[3] > 6.0 and _wcolorful >= 4.0 and _g[5] != "coloured":
        _g[5] = "coloured"
        _g[6] = _g[6] or "natural tile"
        print(f"guard: {_g[0]} is mostly TILE ({100*_tsh:.0f}%) with its own warm "
              f"tone -> COLOURED (tiles keep their natural tone)")
# FLAKINESS GUARDS — measurement overrides a doubtful verdict either way:
for _g in w_groups:
    if _g[4] < 4.5 and _g[5] != "white":
        _g[5] = "white"; _g[6] = ""
        print(f"guard: {_g[0]} measures near-neutral (ch {_g[4]:.1f}) -> WHITE")
    elif _g[4] > 18.0 and not (_g[2] < 5.0 and _g[3] > 0.0) and _g[5] != "coloured":
        # strong colour forces COLOURED — except on the pure-yellow axis (a<5,
        # b>0), where even strong warmth can be heavy tungsten cast: there the
        # LLM verdict stands
        _g[5] = "coloured"
        print(f"guard: {_g[0]} carries strong colour (ch {_g[4]:.1f}) -> COLOURED")
for _g, _tsh in zip(w_groups, _wtile_share):
    print(f"  {_g[0]:8s} {100*_g[1].mean():4.1f}% of image · measured a{_g[2]:+.0f} "
          f"b{_g[3]:+.0f} ch{_g[4]:.0f} · tile {100*_tsh:.0f}% -> {_g[5].upper()}"
          + (f" ({_g[6]})" if _g[6] else ""))

# ---- STEP 3: the HDR — one light chain, then per-wall colour treatment ---------
w_img = _levels_soft(w_img, exclude=w_ex)
w_img = _expose(w_img, target=PARAMS["expose_tgt"], exclude=w_ex)
if "EXPO_MAP" in globals():
    w_img = np.clip(w_img, 0, 1) ** EXPO_MAP[..., None]
w_img = _scurve(w_img, PARAMS["scurve"])
w_img = _sharpen(w_img)
w_u8 = (np.clip(w_img, 0, 1) * 255).astype("uint8")
# every material keeps its fused colour (colour source = the cast-corrected fused)
_wbl = cv2.cvtColor(w_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
_wkeep = np.clip((w_La - 0.04) / 0.08, 0, 1)[..., None]
_wbl[..., 1:] = _wbl[..., 1:] * (1 - _wkeep) + w_lab[..., 1:] * _wkeep
w_u8 = cv2.cvtColor(np.clip(_wbl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

# per-wall brightness: white walls -> 0.82 (AutoHDR white), coloured walls ->
# 0.78 (bright but grounded — AutoHDR keeps coloured walls a touch deeper),
# ceiling -> 0.83. Same median/band-guard/feather math as the proven region lift.
w_adj = np.ones((H, Wd), np.float32)
w_lu = w_u8.astype(np.float32).mean(2) / 255
for _gm, _tgt in ([(_g[1], 0.82 if _g[5] == "white" else 0.78) for _g in w_groups]
                  + [(w_ceil, 0.83)]):
    if _gm.sum() < 0.002 * H * Wd:
        continue
    _cur = float(np.median(w_lu[_gm]))
    _ev = float(np.clip(math.log2(max(_tgt, 1e-3) / max(_cur, 1e-3)), -0.8, 0.8))
    _gma = float(np.clip(np.log(max(_cur * (2 ** _ev), 1e-3)) / np.log(max(_cur, 1e-3)),
                         0.5, 2.4)) if _cur > 1e-3 else 1.0
    _band = np.clip(1 - np.abs(w_lu - _cur) / 0.35, 0, 1)
    _soft = cv2.GaussianBlur(_gm.astype(np.float32), (0, 0), max(4, int(Wd * 0.01))) * _band
    w_adj = w_adj * (1 - _soft) + _gma * _soft
w_u8 = ((np.clip(w_u8.astype(np.float32) / 255, 0, 1) ** w_adj[..., None]) * 255).astype("uint8")

# per-wall COLOUR: white walls whitened fully; coloured walls keep their paint
# (already the maintained fused colour) with a gentle saturation enhance.
_wl3 = cv2.cvtColor(w_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
for _g in w_groups:
    _m3 = cv2.GaussianBlur(_g[1].astype(np.float32), (0, 0), 4)
    if _g[5] == "white":
        _m3 = _m3 * 0.95
        _wl3[..., 1] = _wl3[..., 1] * (1 - _m3) + 128 * _m3
        _wl3[..., 2] = _wl3[..., 2] * (1 - _m3) + 128 * _m3
    else:
        _sat3 = 1.0 + 0.06 * _m3                        # enhanced AS its colour
        _wl3[..., 1] = 128 + (_wl3[..., 1] - 128) * _sat3
        _wl3[..., 2] = 128 + (_wl3[..., 2] - 128) * _sat3
_m3c = cv2.GaussianBlur(w_ceil.astype(np.float32), (0, 0), 4) * 0.95
_wl3[..., 1] = _wl3[..., 1] * (1 - _m3c) + 128 * _m3c   # ceiling always clean white
_wl3[..., 2] = _wl3[..., 2] * (1 - _m3c) + 128 * _m3c
w_u8 = cv2.cvtColor(np.clip(_wl3, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

# shade evening on WHITE walls only (they are JUDGED white — no colour to
# protect inside them): one pass, mask-normalised (halo-proof), contact-shade
# ring kept around dark objects, tile texture excluded.
_wtex = cv2.GaussianBlur(np.abs(w_La - cv2.GaussianBlur(w_La, (0, 0), 3)),
                         (0, 0), max(10, int(Wd * 0.015)))
for _g in w_groups:
    if _g[5] != "white" or _g[1].sum() < 0.002 * H * Wd:
        continue
    _wl4 = cv2.cvtColor(w_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    _Le4 = _wl4[..., 0] / 255
    _dob = (_Le4 < 0.50) & ~w_wallm & ~w_ceil
    _oring = (cv2.dilate(_dob.astype(np.uint8),
                         np.ones((max(9, int(Wd * 0.012) | 1),) * 2, np.uint8)).astype(bool)
              & ~_dob)
    _em = _g[1] & (_wtex < 0.026) & ~_oring & (_Le4 > 0.40) & (_Le4 < 0.92)
    _mf4 = _em.astype(np.float32)
    _sg4 = max(15, int(Wd * 0.015))
    _dn4 = cv2.GaussianBlur(_mf4, (0, 0), _sg4)
    _lo4 = cv2.GaussianBlur(_Le4 * _mf4, (0, 0), _sg4) / np.maximum(_dn4, 1e-4)
    _wr4 = float(np.median(_Le4[_g[1]])) if _g[1].sum() > 1000 else 0.80
    _lo4 = np.where(_dn4 > 0.10, _lo4, _wr4)
    _df4 = np.clip(_wr4 - _lo4, 0, 0.20)
    _me4 = cv2.GaussianBlur(_em.astype(np.float32), (0, 0), 6) * 0.9
    _wl4[..., 0] = np.clip(_Le4 + _df4 * _me4, 0, 1) * 255
    w_u8 = cv2.cvtColor(np.clip(_wl4, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

# window: keep the fused view's brightness (no wash) — same pull as CELL 12
if W_WINDOW.sum() > 0.001 * H * Wd:
    _wl5 = cv2.cvtColor(w_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    _mw5 = cv2.GaussianBlur(W_WINDOW.astype(np.float32), (0, 0), 3) * 0.85
    _wl5[..., 0] = _wl5[..., 0] * (1 - _mw5) + w_lab[..., 0] * _mw5
    w_u8 = cv2.cvtColor(np.clip(_wl5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

w_u8 = (_soft_highlights(w_u8.astype(np.float32) / 255, knee=0.93, slope=0.5)
        * 255).clip(0, 255).astype("uint8")
cv2.imwrite("hdr_result_walls.jpg", w_u8, [cv2.IMWRITE_JPEG_QUALITY, 95])

# show the wall map + the result
_wov = fused.astype(np.float32).copy()
for _g, _col in zip(w_groups, ((255, 120, 0), (0, 0, 230))):
    _a6 = _g[1].astype(np.float32)[..., None] * 0.4
    _t6 = np.zeros_like(_wov); _t6[:] = _col
    _wov = _wov * (1 - _a6) + _t6 * _a6
plt.figure(figsize=(18, 7))
plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(_wov.clip(0, 255).astype("uint8"),
                                              cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title("WALL GROUPS — " + " · ".join(f"{_g[0]}={_g[5]}" + (f"({_g[6]})" if _g[6] else "")
                                        for _g in w_groups))
plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(w_u8, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title("PER-WALL COLOUR HDR — white walls white, coloured walls kept")
plt.tight_layout(); plt.show()
try:
    files.download("hdr_result_walls.jpg")
except Exception:
    pass
print("done — hdr_result_walls.jpg (per-wall colour HDR; CELL 12 is unchanged below)")


# ============================== CELL 12 — FUSED-BASED HDR (segmentation-guided) ==
# THIS CELL WORKS ON THE FUSED IMAGE — not on the CELL-4 global result.
# The fused image is the ground truth: it still holds every material's TRUE
# colour and detail. The new-HDR brightness chain is applied to the FUSED image
# directly and the fused image's COLOURS are the colour source everywhere. Then:
# PAINTED walls -> clean white · TILES / MARBLE -> natural neutral tone (never
# forced white) · the VANITY keeps its wood colour and its fused detail · the
# window view is untouched. OneFormer guides the per-region exposure (a ±EV
# table is printed) and the small local fixes.
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if "of_model" not in globals():                        # already loaded earlier here
    try:
        OF_ID = "shi-labs/oneformer_ade20k_dinat_large"
        of_proc = OneFormerProcessor.from_pretrained(OF_ID)
        of_model = OneFormerForUniversalSegmentation.from_pretrained(OF_ID).to(DEVICE).eval()
    except Exception as e:
        print(f"DiNAT unavailable ({type(e).__name__}) — Swin-L fallback")
        OF_ID = "shi-labs/oneformer_ade20k_swin_large"
        of_proc = OneFormerProcessor.from_pretrained(OF_ID)
        of_model = OneFormerForUniversalSegmentation.from_pretrained(OF_ID).to(DEVICE).eval()

from PIL import Image
# SEGMENT ON THE FUSED IMAGE: the fused image still shows every material's TRUE
# colour — golden vanity wood, marble tile veins, stainless steel — so OneFormer
# separates the classes far more reliably there. ALL of CELL 5's work then
# happens on the fused image itself (built just below) — the CELL-4 global
# result is NOT used as a base here.
pil = Image.fromarray(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
inp = of_proc(images=pil, task_inputs=["semantic"], return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = of_model(**inp)
seg = of_proc.post_process_semantic_segmentation(out, target_sizes=[pil.size[::-1]])[0].cpu().numpy()

# ---- LLM SCENE JUDGE: dynamic decisions instead of hardcoded thresholds ---------
# Every room is different — fixed thresholds (b>18 = wood, 20% blown = glass...)
# always break on the next scene. So the JUDGMENTS are made by a vision LLM that
# LOOKS at the fused image once per scene: is the room white or coloured, what
# each door/cabinet object truly is (glass/painted/wood/vanity), what the floor
# material and tone are. The LLM outputs ONLY a small JSON of decisions — it
# never touches a single pixel (no generation, no hallucination possible); all
# pixel work stays in the deterministic math below. If no API key / no network,
# every built-in heuristic still runs as the automatic fallback.
USE_LLM_JUDGE = True    #@param {type:"boolean"}
GEMINI_API_KEY = ""     #@param {type:"string"}   # empty = use env var / Colab secret / .env
LLM_MODEL = "gemini-pro-latest"   #@param {type:"string"}   # alias = the newest Gemini Pro
_LLM_FALLBACKS = ("gemini-pro-latest", "gemini-3.1-pro-preview", "gemini-3.5-flash",
                  "gemini-flash-latest", "gemini-2.0-flash")
LLM = None
LLM_VANITY = np.zeros((H, Wd), bool)      # wood-vanity components per the LLM
WHITEN_EXTRA = np.zeros((H, Wd), bool)    # white cabinets: whiten colour, keep brightness
LLM_SEEN = np.zeros((H, Wd), bool)        # every component the LLM judged — the heuristic
                                          # below must not second-guess those, but it DOES
                                          # still check the small leftovers the LLM never saw
if USE_LLM_JUDGE:
    try:
        import os as _os, base64 as _b64, json as _js, urllib.request as _ur
        _key = (GEMINI_API_KEY or "").strip() or _os.environ.get("GEMINI_API_KEY", "")
        if not _key:
            try:
                from google.colab import userdata as _ud
                _key = _ud.get("GEMINI_API_KEY") or ""
            except Exception:
                pass
        if not _key and _os.path.exists(".env"):
            for _ln in open(".env"):
                if _ln.startswith("GEMINI_API_KEY="):
                    _key = _ln.split("=", 1)[1].strip()
        # SANITIZE: a pasted key often carries hidden line-breaks/spaces (copied
        # from a wrapped display) — remove ALL whitespace so the URL is valid
        _key = "".join(_key.split())
        if not _key:
            raise RuntimeError("no GEMINI_API_KEY")
        # candidate objects the LLM must judge: door + WINDOW + furniture components
        # (window components too: OneFormer labels a curtain-covered door/opening
        # as 'windowpane', and nothing else ever re-checks the window class — the
        # LLM looks at each one and says whether it is real open glass or fabric)
        _cands = []
        for _classes, _minfrac in (((14,), 0.0005), ((8,), 0.002),
                                   ((10, 24, 44, 35), 0.002), ((15, 33), 0.002)):
            _msk0 = np.isin(seg, _classes)
            if not _msk0.any():
                continue
            _n0, _l0, _s0, _ = cv2.connectedComponentsWithStats(_msk0.astype(np.uint8))
            for _j in range(1, _n0):
                if _s0[_j, cv2.CC_STAT_AREA] >= _minfrac * H * Wd:
                    _cands.append((len(_cands) + 1, _l0 == _j))
        _ann = fused.copy()
        for _cid, _cm in _cands:
            _ys, _xs = np.where(_cm)
            cv2.rectangle(_ann, (int(_xs.min()), int(_ys.min())), (int(_xs.max()), int(_ys.max())),
                          (0, 0, 255), max(2, Wd // 500))
            cv2.putText(_ann, str(_cid), (int(_xs.min()) + 8, int(_ys.min()) + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        _ann = cv2.resize(_ann, (768, int(_ann.shape[0] * 768 / _ann.shape[1])))
        _jpg = cv2.imencode(".jpg", _ann, [cv2.IMWRITE_JPEG_QUALITY, 88])[1]
        # PROOF the image really goes to Gemini: show EXACTLY what is being sent
        plt.figure(figsize=(9, 6))
        plt.imshow(cv2.cvtColor(_ann, cv2.COLOR_BGR2RGB)); plt.axis("off")
        plt.title(f"IMAGE SENT TO GEMINI — {_ann.shape[1]}x{_ann.shape[0]}px · "
                  f"{len(_jpg)/1024:.0f} KB · {len(_cands)} numbered object(s)")
        plt.show()
        print(f"LLM judge: sending the {len(_jpg)/1024:.0f} KB image above to Gemini...")
        _prompt = (
            'You are a real-estate photo analyst. This is the UNEDITED fused photo of a room; '
            'the light may be warm/yellow - judge what the materials TRULY are, ignoring the '
            'colour of the light falling on them. Red numbered boxes mark objects to judge.\n'
            'Return ONLY this JSON:\n'
            '{"scene": "white" or "coloured",\n'
            ' "objects": [{"id": <box number>, "kind": "<kind>"}],\n'
            ' "floor": {"material": "wood"|"tile"|"stone"|"carpet"|"other",'
            ' "tone": "warm_oak"|"light_oak"|"golden"|"grey"|"other"}}\n'
            'Rules:\n'
            '- "scene": judge WITHOUT counting the floor (the floor is handled separately). '
            '"coloured" if any significant real coloured material exists above the floor — '
            'wood furniture, a wooden vanity or wooden cabinets, warm-toned or coloured wall '
            'tile, coloured walls. "white" if every surface above the floor except small '
            'decor items is truly white/neutral.\n'
            '- kinds: "window" (a real window: open glass, the outside/daylight visible '
            'through it), "curtained" (a curtain/blind/fabric covers this opening or door - '
            'no open glass view), "glass_door" (outside visible through it), "painted_door" '
            '(ALSO white louvered/slatted closet doors - slats are NOT window blinds!), '
            '"wood_door", "wood_vanity" (wooden bathroom vanity), "white_cabinet", '
            '"wood_cabinet" (ANY wooden cabinet, dresser or chest of drawers - its wood '
            'colour must be kept), "wood_table" (a real wood '
            'table/desk - its colour must be kept), "painted_panel", "shelf", "other".\n'
            '- give a kind for EVERY numbered box.')
        _body = {"contents": [{"parts": [
                    {"inline_data": {"mime_type": "image/jpeg",
                                     "data": _b64.b64encode(_jpg.tobytes()).decode()}},
                    {"text": _prompt}]}],
                 "generationConfig": {"temperature": 0, "response_mime_type": "application/json"}}
        import time as _tm
        _t0 = _tm.time()
        _resp = None
        _mused = ""
        _lasterr = ""
        for _mname in ((LLM_MODEL,) + tuple(m for m in _LLM_FALLBACKS if m != LLM_MODEL)):
            try:
                _req = _ur.Request(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{_mname}:generateContent?key={_key}",
                    data=_js.dumps(_body).encode(), headers={"Content-Type": "application/json"})
                with _ur.urlopen(_req, timeout=90) as _r:
                    _resp = _js.loads(_r.read())
                _mused = _mname
                break
            except Exception as _me:
                # remember WHY this model failed so the user can actually see it
                try:
                    _lasterr = (f"{_mname}: HTTP {_me.code} — {_me.read()[:300].decode(errors='ignore')}"
                                if hasattr(_me, "code") else f"{_mname}: {type(_me).__name__}: {str(_me)[:200]}")
                except Exception:
                    _lasterr = f"{_mname}: {type(_me).__name__}"
                continue
        if _resp is None:
            raise RuntimeError(f"all models failed — last error: "
                               f"{_lasterr.replace(_key, '[KEY]')}")   # never print the key
        # server-side PROOF the image was received: Gemini's own token count for
        # the request (an image counts hundreds of prompt tokens)
        _um = _resp.get("usageMetadata", {})
        print(f"LLM judge: answer from '{_mused}' in {_tm.time()-_t0:.1f}s — "
              f"prompt tokens {_um.get('promptTokenCount', '?')} (image included), "
              f"answer tokens {_um.get('candidatesTokenCount', '?')}")
        # robust extraction: JOIN every text part — the answer sometimes arrives
        # split across parts (the closing brace in its own piece), and taking
        # only the last part loses the JSON entirely
        _txt = "".join(_p["text"] for _p in _resp["candidates"][0]["content"]["parts"]
                       if isinstance(_p.get("text"), str))
        # show EXACTLY what Gemini returned, word for word
        print("LLM judge — Gemini's raw answer:")
        print(_txt.strip()[:1200])
        # try parsing a complete JSON object starting at each '{' (skips thought
        # text before it; ignores stray braces/text after it)
        _dec = _js.JSONDecoder()
        LLM = None
        _pos = _txt.find("{")
        while _pos != -1:
            try:
                _obj = _dec.raw_decode(_txt[_pos:])[0]
                if isinstance(_obj, dict) and "scene" in _obj:
                    LLM = _obj
                    break
            except Exception:
                pass
            _pos = _txt.find("{", _pos + 1)
        if LLM is None and "{" in _txt:
            # LAST RESORT — the answer itself was truncated (missing closing
            # braces): complete the braces and try once more
            _fix = _txt[_txt.find("{"):]
            _fix = _fix + "}" * max(0, _fix.count("{") - _fix.count("}"))
            try:
                _obj = _dec.raw_decode(_fix)[0]
                if isinstance(_obj, dict) and "scene" in _obj:
                    LLM = _obj
                    print("(truncated answer repaired by completing the braces)")
            except Exception:
                pass
        if LLM is None:
            raise RuntimeError("no valid JSON object found in the answer")
        _kinds = {int(o.get("id", -1)): str(o.get("kind", "")) for o in LLM.get("objects", [])}
        # consistency: a room containing real wood objects can never be a "white" scene
        LLM["has_wood"] = any(_k in ("wood_vanity", "wood_cabinet", "wood_door", "wood_table")
                              for _k in _kinds.values())
        if LLM["has_wood"]:
            LLM["scene"] = "coloured"
        # apply the verdicts: pixels are only RE-LABELLED — never changed here.
        # EVIDENCE VETO on window verdicts: a real window / glass door shows
        # DAYLIGHT — it is partly blown in the pre-recovery fusion. A white
        # louvered closet door LOOKS like window shutters to a vision model,
        # but it has no daylight behind it — physics beats the verdict.
        _vlu = (mertens if "mertens" in globals() else fused).astype(np.float32).mean(2) / 255
        _fbv2 = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 2] - 128
        _scb2 = float(np.median(_fbv2))
        for _cid, _cm in _cands:
            LLM_SEEN |= _cm
            _k = _kinds.get(_cid, "")
            if _k in ("window", "glass_door") and float((_vlu[_cm] > 0.88).mean()) < 0.05:
                # real WOOD is ABSOLUTELY warm (fused b > 18) — merely warmer than
                # the scene is bounce-cast on white paint and must be whitened
                _k = ("wood_door" if (float(np.median(_fbv2[_cm])) - _scb2 > 8
                                      and float(np.median(_fbv2[_cm])) > 18)
                      else "painted_door")
                print(f"evidence veto: box #{_cid} was called a window but shows NO "
                      f"daylight -> treated as {_k} (louvered/closet door)")
            if _k in ("painted_door", "painted_panel"):
                seg[_cm] = 0                       # whitened + exposed like the wall
            elif _k == "wood_door":
                seg[_cm] = 200                     # material colour kept, not a window
            elif _k == "curtained":
                seg[_cm] = 200                     # fabric-covered opening — NOT a window:
                                                   # no window-pull on the curtain; it
                                                   # follows the normal room treatment
            elif _k == "wood_vanity":
                LLM_VANITY |= _cm                  # grounding + fused-detail treatment
            elif _k == "white_cabinet":
                WHITEN_EXTRA |= _cm                # whiten colour, keep its brightness
            elif (_k not in ("window", "glass_door")
                  and float(np.isin(seg[_cm], (8, 14)).mean()) > 0.5):
                # a door/window-labelled box with an UNRECOGNISED verdict ('other',
                # typo, ...) must not silently stay a window — the daylight
                # evidence decides it directly
                if float((_vlu[_cm] > 0.88).mean()) < 0.05:
                    # painted-vs-wood decided here too: a blanket "keep colour"
                    # left a bounce-warmed white door jamb yellow (measured b+21)
                    _wd_o = (float(np.median(_fbv2[_cm])) - _scb2 > 8
                             and float(np.median(_fbv2[_cm])) > 18)
                    seg[_cm] = 200 if _wd_o else 0
                    print(f"box #{_cid} ('{_k}') sits on a door/window label with no "
                          f"daylight -> " + ("wood: colour kept" if _wd_o else
                          "painted: whitened like the wall") + " (not a window)")
            # glass_door stays window; wood_cabinet / shelf / other stay untouched
        print(f"LLM judge: scene={LLM.get('scene')} · "
              + " · ".join(f"#{i}={k}" for i, k in sorted(_kinds.items()))
              + f" · floor={(LLM.get('floor') or {}).get('material')}/{(LLM.get('floor') or {}).get('tone')}")
    except Exception as _e:
        LLM = None
        print(f"LLM judge unavailable ({type(_e).__name__}: {str(_e)[:300]}) — "
              f"using the built-in heuristics")

# ---- GLASS door vs SOLID door: ADE20K class 14 is ANY door -----------------------
# A GLASS door (patio/french door showing the outside) must be treated like a
# window. But a SOLID interior door — or a mirrored door — is just a room
# surface: treating it as "window" wrongly shields it from the room lifts and
# the window-pull would darken it. The evidence: a real glass door is BLOWN
# bright in the PRE-recovery fusion (daylight through the glass, like every
# window); a solid door is no brighter than the room. Solid doors are
# reclassified to a neutral id (200): not window, not wall — just an object
# whose colour and brightness follow the normal image treatment.
_door = np.isin(seg, (8, 14)) & ~LLM_SEEN                  # doors AND window-labelled areas —
                                                           # a curtained opening often reads
                                                           # 'windowpane' and must be re-checked.
                                                           # LLM-judged components are settled;
                                                           # the SMALL leftovers it never saw
                                                           # (a transom sliver above a door)
                                                           # still get the evidence test below
if _door.any():
    _pre = mertens if "mertens" in globals() else fused    # pre-recovery: windows blow out there
    _plu = _pre.astype(np.float32).mean(2) / 255
    _fbd = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 2] - 128
    _scene_b = float(np.median(_fbd))                      # the scene's own cast level
    # a glass door must also show a VIEW: the recovered fused content has real
    # structure (mullions, trees, outside detail). A SUNLIT white door also blows
    # out in the pre-recovery fusion, but it recovers to smooth flat paint — the
    # view-texture test tells the two apart.
    _fld = fused.astype(np.float32).mean(2) / 255
    _texd = cv2.GaussianBlur(np.abs(_fld - cv2.GaussianBlur(_fld, (0, 0), 3)),
                             (0, 0), max(10, int(Wd * 0.015)))
    _nd, _ld, _sd, _ = cv2.connectedComponentsWithStats(_door.astype(np.uint8))
    for _i in range(1, _nd):
        if _sd[_i, cv2.CC_STAT_AREA] < 0.0002 * H * Wd:    # even small slivers checked
            continue
        _dc = _ld == _i
        _is_glass = (float((_plu[_dc] > 0.88).mean()) >= 0.20     # blown daylight AND
                     and float(np.median(_texd[_dc])) >= 0.024)   # a real view behind it
        if not _is_glass:
            if float((seg[_dc] == 8).mean()) > 0.5:        # a 'window' with NO real view
                seg[_dc] = 200                             # behind it: fabric/curtain covers
                print(f"curtained 'window' ({100*_dc.mean():.1f}% of image) — fabric, not "
                      f"open glass -> normal room treatment (no window-pull)")
                continue
            # -> SOLID door.
            # PAINTED door (white/grey — no warmer than the scene's own cast) is a
            # wall surface: it joins the wall class and gets whitened like the
            # paint around it (AutoHDR: white door b+1). A WOOD door keeps its
            # material colour (neutral id — no window, no whitening).
            _rel_b = float(np.median(_fbd[_dc])) - _scene_b
            # real WOOD is ABSOLUTELY warm (fused b > 18); a white door/jamb that
            # merely picked up warm bounce light stays a painted surface
            _is_wood_d = _rel_b > 8 and float(np.median(_fbd[_dc])) > 18
            seg[_dc] = 200 if _is_wood_d else 0
            print(f"solid door detected ({100*_dc.mean():.1f}% of image) — "
                  + ("WOOD door: material colour kept" if _is_wood_d
                     else "painted door: whitened like the wall") + " (not a window)")

# ---- THE FUSED-BASED HDR: the new-HDR brightness chain applied to the FUSED image
# Only the LIGHT is processed — levels, room exposure, the CELL-3 analysis dodge,
# gentle contrast, sharpen, soft highlights. Every COLOUR-changing step of the
# global chain (neutralize / match-white / lift-whites / desat / pinch) is
# DELIBERATELY LEFT OUT: the colours are taken from the fused image itself below,
# so every material keeps its true colour (the "maintain the colours" rule).
img5 = fused.astype(np.float32) / 255
ex5 = _bright_ramp(img5)                       # shield the window/lights from the lifts
img5 = _wb(img5, exclude=ex5)                  # GLOBAL cast fix only (one gain per
                                               # channel): removes the tungsten tint,
                                               # keeps every material's colour identity
fusedC = (np.clip(img5, 0, 1) * 255).astype("uint8")   # <- the COLOUR + DETAIL SOURCE
img5 = _levels_soft(img5, exclude=ex5)         # highlights maintained
img5 = _expose(img5, target=PARAMS["expose_tgt"], exclude=ex5)   # room lift, no wash-out
if "EXPO_MAP" in globals():                    # CELL-3 analysis DRIVES the exposure:
    img5 = np.clip(img5, 0, 1) ** EXPO_MAP[..., None]   # dark zones lifted, blacks kept
img5 = _scurve(img5, PARAMS["scurve"])
img5 = _sharpen(img5)
# NO highlight compression here — the single anti-clip pass runs ONCE at the very
# end of the cell. Stacking three compressions (here + sun-tone + final safety)
# was flattening every bright wall into one milky tone: measured wall light
# spread 0.005 vs AutoHDR's 0.043 — the natural light gradient was being erased.
base = (np.clip(img5, 0, 1) * 255).astype("uint8")

# ---- MAINTAIN THE FUSED COLOURS: every pixel's colour (LAB a,b) is put back from
# the cast-corrected fused image, at the new HDR brightness. Walls keep their real
# paint colour (NOT forced white), the vanity keeps its wood, tiles keep their
# marble veins — nothing is neutralised or bleached. (Only near-black fused pixels
# keep the base colour — their fused colour is just sensor noise.)
MAINTAIN_COLORS = 1.0   #@param {type:"number"}   # 1 = full fused colour (default), 0 = off
if MAINTAIN_COLORS > 0:
    b_lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    f_lab = cv2.cvtColor(fusedC, cv2.COLOR_BGR2LAB).astype(np.float32)
    _fL = f_lab[..., 0] / 255
    _keep = (MAINTAIN_COLORS * np.clip((_fL - 0.04) / 0.08, 0, 1))[..., None]
    b_lab[..., 1:] = b_lab[..., 1:] * (1 - _keep) + f_lab[..., 1:] * _keep
    base = cv2.cvtColor(np.clip(b_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"fused colours maintained over the whole image (MAINTAIN_COLORS={MAINTAIN_COLORS})")

# ---- WHITE-SCENE AUTO-CLEAN (AutoHDR's measured behaviour) ---------------------
# AutoHDR decides PER SCENE: in the white kitchen it neutralises everything
# (measured: cabinet b+0, wall b+0, floor b+1 — the scene's warmth is all lighting
# cast, there is no real colour), while in the bathroom it MAINTAINS the colours
# (wall b+11 kept, vanity wood b+26 kept). Measured on the fused images, the two
# are separable by how much REAL strong colour the scene contains (white kitchen
# 2.1% vs vanity bathroom 5.4%). So: a scene with (almost) no real colour is a
# WHITE scene -> its low-chroma cast is cleaned; any scene with real coloured
# materials is left fully maintained. Strong colour (chroma > 26) NEVER moves.
_base_warm = base                # snapshot BEFORE the cleaning below — the bulb/glow
                                 # detection further down must see the ORIGINAL warmth
                                 # (after cleaning, a warm bulb reads neutral and would
                                 # be missed in a white scene)
_lab_c = cv2.cvtColor(fusedC, cv2.COLOR_BGR2LAB).astype(np.float32)
_Lc = _lab_c[..., 0] / 255
_ac = _lab_c[..., 1] - 128; _bc = _lab_c[..., 2] - 128
_chc = np.sqrt(_ac * _ac + _bc * _bc)
_win_e = np.isin(seg, (8, 14))                             # the window view's colour
_valid = (_Lc > 0.15) & (_Lc < 0.95) & ~_win_e             # does not count as scene colour
# HARD FURNITURE is real material, never "lighting cast": a wood table/desk/chair
# must NEVER be bleached by the white-scene clean — its colour is its identity.
# (A table's wood is often chroma 15-25, exactly the range the cast clean works
# in — without this exclusion a real table turns white in a white scene.)
_furn_e = np.isin(seg, (15, 33, 19, 30))                   # table, desk, chair, armchair
_colorful = float((_chc[_valid] > 25).mean()) * 100 if _valid.sum() > 1000 else 100.0
white_scene = float(np.clip((5.0 - _colorful) / 2.5, 0, 1))
if LLM is not None and LLM.get("scene") in ("white", "coloured"):
    white_scene = 1.0 if LLM["scene"] == "white" else 0.0   # the LLM's verdict decides
    # FLAKINESS GUARD: a vision answer can vary between runs on the same photo.
    # When the measurement is unambiguous — almost no real colour anywhere (a
    # white kitchen measures ~2%) and the LLM itself saw no wood object — the
    # scene IS white, whatever the scene word said this time.
    if white_scene == 0.0 and _colorful < 3.0 and not LLM.get("has_wood"):
        white_scene = 1.0
        print("scene guard: <3% real colour and no wood objects -> WHITE scene "
              "(overriding a flaky 'coloured' verdict)")
if white_scene > 0:
    b_lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    _a2 = b_lab[..., 1] - 128; _b2 = b_lab[..., 2] - 128
    _ch2 = np.sqrt(_a2 * _a2 + _b2 * _b2)
    _gate = np.clip((26 - _ch2) / 8, 0, 1) * white_scene   # low-chroma cast only;
    # cast lives on BRIGHT surfaces (white paint/cabinets under warm light). A
    # DARK pixel's colour is MATERIAL — a walnut dresser, a bed frame — never
    # lighting cast, so the clean fades out completely below L 0.45.
    _gate = _gate * np.clip((b_lab[..., 0] / 255 - 0.45) / 0.15, 0, 1)
    _gate = _gate * (1 - (_win_e | _furn_e).astype(np.float32))   # real colour + window
                                                           # + hard furniture (table!) stay
    # NOT blurred on purpose: the chroma ramp is already continuous (no seams),
    # and a spatial blur would leak the cleaning onto the rim of every coloured
    # object and into the window view — exactly the protections above.
    b_lab[..., 1] = _a2 * (1 - _gate) + 128
    b_lab[..., 2] = _b2 * (1 - _gate) + 128
    base = cv2.cvtColor(np.clip(b_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
print(f"scene colour check: {_colorful:.1f}% real colour -> "
      + ("WHITE scene — lighting cast cleaned to white (AutoHDR behaviour)" if white_scene > 0.5
         else ("mixed scene — cast partially cleaned" if white_scene > 0
               else "COLOURED scene — colours fully maintained (AutoHDR behaviour)"))
      + f"  (white_scene={white_scene:.2f})")

# ---- TAME THE MAINTAINED CAST (calibrated on AutoHDR) --------------------------
# Maintaining the fused colours keeps a little too much of the warm LIGHTING cast
# on the low/mid-chroma surfaces. Measured on the bathroom vs AutoHDR: our walls
# b+4 vs their -1, wainscot tiles +7 vs +1, shower tiles +18 vs +12, mid-tone
# warmth +10 vs +1 — while the wood vanity is already right (+24 vs +25). So the
# cast is cooled ONLY on cast-carrying surfaces (chroma ramp below 26), only where
# the pixel actually IS warm, never past neutral (no blue shift), and strong real
# colour (the vanity wood, decor) never moves. In a WHITE scene the auto-clean
# above has already zeroed the cast, so this step is a natural no-op there.
CAST_TAME = 5.0   #@param {type:"number"}   # 0 = off · 5 = AutoHDR-calibrated (default)
if CAST_TAME > 0:
    b_lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    _a3 = b_lab[..., 1] - 128; _b3 = b_lab[..., 2] - 128
    _ch3 = np.sqrt(_a3 * _a3 + _b3 * _b3)
    _w3 = np.clip((26 - _ch3) / 12, 0, 1) * np.clip(_b3 / 4, 0, 1)
    _w3 = _w3 * (1 - (_win_e | _furn_e).astype(np.float32))   # window untouched; hard
                                                           # furniture keeps its warmth too
    b_lab[..., 2] = np.where(_b3 > 0, np.maximum(_b3 - CAST_TAME * _w3, 0.0), _b3) + 128
    b_lab[..., 1] = np.where(_a3 > 0, np.maximum(_a3 - 0.35 * CAST_TAME * _w3, 0.0), _a3) + 128
    base = cv2.cvtColor(np.clip(b_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"cast tame {CAST_TAME}: warm lighting cast cooled to AutoHDR level (real colour kept)")

# ---- FIND THE VANITY (wood ONLY — not cabinets/shelves/wardrobes) --------------
# The grounding + detail treatment further down is for a REAL WOOD VANITY only.
# Kitchen cabinets, shelves and wardrobes are NOT included — a white kitchen
# cabinet must stay bright white (AutoHDR keeps it at L0.84). A vanity is a
# cabinet-class OBJECT whose fused colour is genuinely warm wood (median fused
# b > 18 per component — measured: wood vanity +25, white kitchen cabinet +16);
# in a WHITE scene there is no vanity by definition.
_vb5 = cv2.cvtColor(fusedC, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 2] - 128
_cab = np.isin(seg, (10, 24, 44, 35)) & ~np.isin(seg, (8, 14))
VANITY = np.zeros((H, Wd), bool)
# PAINTED-PANEL catch-all: OneFormer often labels a white panel DOOR as
# 'wardrobe' or 'cabinet'. Such a surface then got NO treatment at all (not
# wall -> not whitened; not warm wood -> not vanity) and stayed cream with the
# maintained cast. So: a furniture-class object that is SMOOTH (no wood grain /
# tile texture in the fused image) and NO WARMER than the scene's own cast is
# painted trim — it joins the WALL class and gets whitened like the paint.
_fl_v = fused.astype(np.float32).mean(2) / 255
_tex_v = cv2.GaussianBlur(np.abs(_fl_v - cv2.GaussianBlur(_fl_v, (0, 0), 3)),
                          (0, 0), max(10, int(Wd * 0.015)))
_fb_v = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 2] - 128
_scene_bv = float(np.median(_fb_v))
if LLM is not None:                                    # the LLM already judged every object
    VANITY = LLM_VANITY.copy()
elif _cab.any():
    _nv, _lv, _sv, _ = cv2.connectedComponentsWithStats(_cab.astype(np.uint8))
    for _i in range(1, _nv):
        if _sv[_i, cv2.CC_STAT_AREA] < 0.002 * H * Wd:
            continue
        _c = _lv == _i
        if white_scene < 0.5 and float(np.median(_vb5[_c])) > 18:   # warm wood -> VANITY
            VANITY |= _c
        elif (white_scene < 0.5                                     # (white scenes: auto-clean
              and float(np.median(_tex_v[_c])) < 0.024              #  already handles them)
              and float(np.median(_fb_v[_c])) - _scene_bv <= 8):    # smooth + not warmer than cast
            seg[_c] = 0                                             # -> whitened like the wall
            print(f"painted panel/door ({100*_c.mean():.1f}% of image, was furniture-class) "
                  f"-> whitened like the wall")
print(f"wood VANITY found on {100*VANITY.mean():.1f}% of the image" if VANITY.any()
      else "no wood vanity in this scene — cabinets/shelves stay untouched")

# ---- SHOW the segmentation as a colour mask so you can SEE what was detected ---- 
# each region gets its own colour tinted over the result; % of image printed.
SEG_COLOURS = {                                        # name: (classes, BGR colour)
    "wall/ceiling": ((0, 5, 1),         (255, 120, 0)),   # blue
    "floor/rug":    ((3, 28, 13),       (0, 200, 0)),     # green
    "sink/basin":   ((47, 37, 65),      (140, 90, 255)),  # pink — sink, bathtub, toilet
    "appliance":    ((71, 50, 118, 124, 129, 107, 133), (200, 0, 200)),  # magenta — stove/fridge/oven/…
    "window/door":  ((8, 14),           (255, 255, 0)),   # cyan
    "lamp/fan":     ((36, 82, 85, 139), (0, 140, 255)),   # orange — lamp, light, chandelier, fan
}
# The wood VANITY (found above) is shown in RED in the mask view. Kitchen
# cabinets / shelves / wardrobes are deliberately NOT treated as vanity — they
# get no grounding and no detail transplant, only the normal colour handling.
# ---- BULBS + their GLOW halo (the warm reflection spreading around a light) ----
# OneFormer's lamp classes catch fixtures; a bright + warm blob catches bulbs it
# missed (recessed cans, pendant bulbs). GLOW = a dilated halo around them that
# actually landed on the white ceiling/wall — that is the warm reflection we clean.
_lb = cv2.cvtColor(_base_warm, cv2.COLOR_BGR2LAB).astype(np.float32)   # PRE-clean warmth —
_Lr = _lb[..., 0] / 255; _ar = _lb[..., 1] - 128; _brr = _lb[..., 2] - 128  # bulbs stay findable
_chr = np.sqrt(_ar * _ar + _brr * _brr)
# bright-warm blobs: keep only SMALL components (recessed cans / pendant bulbs).
# A LARGE bright-warm area is a sunlit material (counter, curtain, warm wall) —
# with the colours maintained it looks "bulb-warm" here, but flagging it would
# wrongly shield it from SUN_TONE below.
_blob = (_Lr > 0.86) & (_brr > 5) & ~np.isin(seg, (36, 82, 85))
# EMITTER test — catch the BULB, not its reflection: a real bulb EMITS light, so
# it stays bright even in the DARKEST bracket (-EV); a warm reflection on the
# ceiling/wall/tile is only bounced light and goes dark there. Measured: real
# bulbs/recessed cans 0.52-0.83 in the darkest bracket, reflections 0.46-0.48
# -> the 0.50 threshold splits them cleanly.
_dklu = (aligned[0] if "aligned" in globals() else fused).astype(np.float32).mean(2) / 255
_nb, _lbc, _stb, _ = cv2.connectedComponentsWithStats(_blob.astype(np.uint8))
_small = np.zeros(_blob.shape, bool)
_refl = 0
for _i in range(1, _nb):
    if _stb[_i, cv2.CC_STAT_AREA] > 0.0035 * H * Wd:
        continue
    _c = _lbc == _i
    _med_dk = float(np.median(_dklu[_c]))
    _med_L = float(np.median(_Lr[_c])); _med_b = float(np.median(_brr[_c]))
    # tier 1: still bright at -EV -> a bare bulb. tier 2: a DIFFUSED light (fan
    # dome, lantern, glowing shade) spreads its light through glass/fabric so it
    # is dimmer at -EV — but it is a SUBSTANTIAL, very bright, very warm blob
    # that STANDS OUT against its -EV surroundings (a glowing dome on a ceiling
    # that went dark). A blown warm patch (mirror sheen, window-lit tile) is as
    # bright NOW, but at -EV it sits inside an equally-bright area — no stand-out.
    _t2 = (_stb[_i, cv2.CC_STAT_AREA] >= 0.0002 * H * Wd
           and _med_dk > 0.38 and _med_L > 0.92 and _med_b > 8)
    if _t2 and _med_dk <= 0.50:                        # ring test only when needed
        _x0 = max(0, _stb[_i, cv2.CC_STAT_LEFT] - 20)
        _y0 = max(0, _stb[_i, cv2.CC_STAT_TOP] - 20)
        _x1 = min(Wd, _stb[_i, cv2.CC_STAT_LEFT] + _stb[_i, cv2.CC_STAT_WIDTH] + 20)
        _y1 = min(H, _stb[_i, cv2.CC_STAT_TOP] + _stb[_i, cv2.CC_STAT_HEIGHT] + 20)
        _cc = _c[_y0:_y1, _x0:_x1]
        _rg = cv2.dilate(_cc.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~_cc
        _t2 = _rg.any() and _med_dk - float(np.median(_dklu[_y0:_y1, _x0:_x1][_rg])) > 0.15
    if _med_dk > 0.50 or _t2:
        _small |= _c
    else:
        _refl += 1                                     # bounced light -> NOT a bulb
if _refl:
    print(f"{_refl} warm reflection(s) rejected by the emitter test (only real bulbs kept)")
# the semantic lamp classes label the WHOLE fixture (arm, chain, shade — the full
# chandelier/fanush) — but only the GLOWING BULB must be caught. The same emitter
# test picks out just the lit part: bright NOW and still bright in the darkest
# bracket. The fixture body stays an ordinary object (colours maintained).
_lamp_emit = np.isin(seg, (36, 82, 85, 139)) & (_Lr > 0.70) & (_dklu > 0.30)  # gentle gates:
                                        # a lit lantern / glowing shade is dimmer than a
                                        # bare bulb but still the LIT part of a lamp —
                                        # safe here because these pixels are semantically
                                        # LAMP (reflections are never lamp-labelled)
BULB = _lamp_emit | _small                        # lit bulbs ONLY — never the fixture body
BULB = cv2.morphologyEx(BULB.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
# WIDE search radius (11% of width): a pendant in front of a wall throws its glow
# far beyond a small fixed ring. The ring finds its own true edge — the
# relative-warmth gate below ends it exactly where the glow ends, so the wide
# radius can never bleach anything that is not genuinely glow-warm.
_gk = max(9, int(Wd * 0.11) | 1)
GLOW = cv2.dilate(BULB.astype(np.uint8), np.ones((_gk, _gk), np.uint8)).astype(bool)
# the ring only exists where the warmth GENUINELY exceeds the ceiling/wall's own
# level (+4) — a neutral LED dome on a neutral ceiling gets NO glow blob at all
# (only the bulb is caught), while a warm pendant's real halo is still found.
_wc_ref = np.isin(seg, (0, 5, 1))
_b_ref = float(np.median(_brr[_wc_ref])) if _wc_ref.sum() > 1000 else 0.0
# glow-eligible SURFACES: walls/ceiling + the LAMP classes + solid doors. OneFormer
# often labels the bright glow region AROUND a pendant as 'lamp' too — those pixels
# got NO treatment at all and stayed a warm patch beside the fixture. Including the
# lamp label here is safe: the dark fixture body fails the brightness gate (L>0.60)
# and a coloured shade / wood fails the chroma gate (<18) — only real glow cleans.
_glow_surf = _wc_ref | np.isin(seg, (36, 82, 85, 139, 200))
# chroma gate: on WALLS strong colour (chroma >= 18) stays protected as real
# material (a tan curtain near a sconce is never bleached) — but ON the lamp
# label itself strong warm IS the glow: a pendant's halo measures chroma 20-30.
# The fixture body is dark (fails L > 0.60) and a genuinely coloured glass shade
# is stronger still (> 32), so raising the limit there only cleans real glow.
_chr_ok = np.where(np.isin(seg, (36, 82, 85, 139)), _chr < 32, _chr < 18)
GLOW = GLOW & ~BULB & (_Lr > 0.60) & (_brr > _b_ref + 4) & _chr_ok & _glow_surf
# ---- BULB-ONLY MODE (user rule): catch ONLY the bulbs themselves — the glow
# ring around them is NOT caught and NOT cleaned. Set True to bring glow back.
CATCH_GLOW = False   #@param {type:"boolean"}
if not CATCH_GLOW:
    GLOW = np.zeros((H, Wd), bool)
    print("bulb-only mode: glow area NOT caught (CATCH_GLOW = False)")

_ov = fused.astype(np.float32).copy()          # overlay on the PURE FUSED image — the
                                               # exact image the segmentation ran on, so
                                               # the mask view can never look like it was
                                               # made from "another image"
print("segments detected:")
for _nm, (_cls, _col) in SEG_COLOURS.items():
    _mk = np.isin(seg, _cls)
    if _mk.mean() < 0.001:
        print(f"  {_nm:14s}  (not found)"); continue
    _a = _mk.astype(np.float32)[..., None] * 0.45
    _t = np.zeros_like(_ov); _t[:] = _col
    _ov = _ov * (1 - _a) + _t * _a
    print(f"  {_nm:14s}  {100*_mk.mean():4.1f}% of image")
for _nm, _msk, _col, _al in (("vanity (wood)", VANITY, (0, 0, 230), 0.45),  # red
                             ("glow halo", GLOW, (0, 220, 255), 0.4),
                             ("bulbs", BULB, (0, 255, 255), 0.6)):   # yellow
    if _msk.mean() < 0.0001:                       # low bar: ONE small fan dome must
                                                   # still show as found, not hide

        print(f"  {_nm:14s}  (not found)"); continue
    _a = _msk.astype(np.float32)[..., None] * _al
    _t = np.zeros_like(_ov); _t[:] = _col
    _ov = _ov * (1 - _a) + _t * _a
    print(f"  {_nm:14s}  {100*_msk.mean():4.2f}% of image")
_ov = _ov.clip(0, 255).astype("uint8")
cv2.imwrite("segmentation_mask.jpg", _ov, [cv2.IMWRITE_JPEG_QUALITY, 92])
plt.figure(figsize=(18, 11))
plt.imshow(cv2.cvtColor(_ov, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title("SEGMENTATION on the PURE FUSED image — blue wall/ceiling · green floor · red vanity(wood) · pink sink · magenta appliance · cyan window · orange lamp/fan · yellow bulb+glow")
plt.show()

# ADE20K classes -> regions, each with a natural luminance target.
# WALL and CEILING are SPLIT on purpose — measured on the AutoHDR reference the
# ceiling sits at ~0.83 but the walls at only ~0.77; one combined 0.82 target
# pushed the walls to ~0.88 (too bright) and drove the blown-highlight count up.
# FLOOR target is ADAPTIVE: floors differ hugely by material (dark walnut vs a
# light marble hex tile), so one fixed number is wrong for most scenes — the old
# fixed 0.52 (calibrated on a dark-wood floor) was darkening light floors by
# almost half an EV. AutoHDR keeps the floor's own level, slightly deepened:
# measured bathroom hex floor 0.71 -> AutoHDR 0.65, light-oak kitchen 0.75 ->
# 0.66, dark wood ~0.52. target = 90% of the floor's own level, clamped.
_floor_cls = np.isin(seg, (3, 28, 13)) & ~np.isin(seg, (8, 14))
_floor_cur = float(np.median(base.astype(np.float32).mean(2)[_floor_cls] / 255)) if _floor_cls.sum() > 1000 else 0.6
FLOOR_TARGET = float(np.clip(_floor_cur * 0.90, 0.45, 0.68))

REGIONS = {
    "ceiling":      {"cls": (5,),             "target": 0.83},   # ceiling
    "wall":         {"cls": (0, 1),           "target": 0.82},   # wall — AutoHDR's brightest
                                                                 # white walls sit at 0.82; this
                                                                 # is the full "more white" look
    "floor":        {"cls": (3, 28, 13),      "target": FLOOR_TARGET},   # adaptive (see above)
    # (the TABLE has NO exposure target on purpose — tables are not re-exposed or
    #  treated at all; they follow the normal image. The table mask below exists
    #  ONLY to protect them from the floor/metal/wall fixes.)
    # A white sink / basin / tub is a WHITE OBJECT, not a wall: pushed to the wall
    # target it blows out and loses its shape. A slightly LOWER target keeps its
    # form and detail (this is the "sink area too bright" fix).
    "sink/basin":   {"cls": (47, 37, 65),     "target": 0.76},   # sink, bathtub, toilet
    # MATERIAL surfaces stay GROUNDED — measured on AutoHDR: the wood vanity sits
    # at L0.62 and the stone counter at L0.68 while the painted walls are 0.78+.
    # Over-brightening these is exactly what washes their detail out and makes
    # the wood look pale ("details are missing"). VANITY ONLY — the wood-vanity
    # mask found above; kitchen cabinets/shelves/wardrobes are NOT grounded.
    "vanity (wood)": {"mask": VANITY,          "target": 0.63},
    "counter":      {"cls": (45,),            "target": 0.68},   # countertop
}
WINDOW = np.isin(seg, (8, 14))                          # window/glass door -> never touched
resf = base.astype(np.float32) / 255                    # <- the FUSED-BASED HDR (not the global result)
lu = resf.mean(2)

print("region              current   target   suggested")
# GLASS-PANE GUARD for the vanity lift: a glass-front cabinet judged 'vanity'
# must brighten its WOOD only. The dark glass panes are COOLER than the wood
# (fused pane b+22 vs oak b+30+) and hold the milky ceiling-light reflections —
# if they ride the +EV lift, those reflections bloom into white halos around
# the glassware inside. Pixels well below the vanity's own wood warmth fade out.
_van_bf = cv2.cvtColor(fusedC, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 2] - 128
_van_gate = np.ones((H, Wd), np.float32)
if VANITY.sum() > 1000:
    _medb_v = float(np.median(_van_bf[VANITY]))
    _van_gate = np.clip((_van_bf - (_medb_v - 8)) / 8, 0, 1)
adjust = np.ones((H, Wd), np.float32)                   # per-pixel gamma map (1 = no change)
for name, spec in REGIONS.items():
    m = (spec["mask"] if "mask" in spec else np.isin(seg, spec["cls"])) & ~WINDOW
    if m.sum() < 0.002 * H * Wd:
        print(f"  {name:16s}  (not found)"); continue
    cur = float(np.median(lu[m])); tgt = spec["target"]
    ev = math.log2(max(tgt, 1e-3) / max(cur, 1e-3))     # stops to move
    ev = float(np.clip(ev, -0.8, 0.8))                  # gentle — never a big jump
    # gamma clamp widened (1.6 -> 2.4): for a BRIGHT region even a small -EV needs
    # a large gamma, and 1.6 was silently stopping the wall from reaching 0.77
    gamma = float(np.clip(np.log(max(cur*(2**ev), 1e-3)) / np.log(max(cur, 1e-3)), 0.5, 2.4)) if cur > 1e-3 else 1.0
    # LUMINANCE-BAND guard: only pixels near the region's median move; a dark
    # shadow corner inside the same mask (far from the median) is left alone,
    # so a strong wall gamma can never crush it. Continuous in luma -> no seams.
    band = np.clip(1 - np.abs(lu - cur) / 0.35, 0, 1)
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(4, int(Wd*0.01))) * band
    if name == "vanity (wood)":
        soft = soft * _van_gate                         # glass panes don't ride the wood lift
    adjust = adjust * (1 - soft) + gamma * soft         # feathered so no region edge shows
    print(f"  {name:16s}   {cur:.2f}     {tgt:.2f}    {ev:+.2f} EV")

local = np.clip(resf, 0, 1) ** adjust[..., None]
local_u8 = (local * 255).clip(0, 255).astype("uint8")

# ---- MAINTAIN THE FUSED DETAIL on the VANITY and the SINK ----------------------
# The brightness chain can flatten the texture of a bright vanity front or a white
# sink. Here the fused image's DETAIL (its high-frequency luminance — wood grain,
# door edges, carving, the sink's shape) is transplanted back at the NEW
# brightness: LOW frequency from the HDR result, HIGH frequency from the fused
# image. VANITY ONLY (the wood-vanity mask) plus counter + sink/tub/toilet —
# kitchen cabinets, shelves and wardrobes are NOT included.
DETAIL_KEEP = 0.8   #@param {type:"number"}   # 0 = off, 0.8 = default, 1 = full fused detail
DETAIL_CLASSES = (45, 47, 37, 65,                   # counter + sink, bathtub, toilet
                  36, 82, 85, 139)                  # + lamp/light/chandelier/FAN fixtures:
                                                    # the HDR lifts wash a fixture out
                                                    # (grey blades, flat shades) — the fused
                                                    # image holds its true crisp contrast,
                                                    # transplanted back at the new brightness
# + the BULBS with their FIXTURE bar: AutoHDR keeps each bulb crisp — bright
# glass on a DARK bar (measured bar darks 0.61 vs our washed-grey 0.71). The
# transplant below restores exactly that fused contrast; the bulb mask is
# dilated so the fixture bar the bulbs sit on is covered too.
_bulbfix = cv2.dilate(BULB.astype(np.uint8),
                      np.ones((max(5, int(Wd * 0.012)) | 1,) * 2, np.uint8)).astype(bool)
det = (np.isin(seg, DETAIL_CLASSES) | (VANITY & (_van_gate > 0.5)) | _bulbfix) & ~WINDOW
#                                      ^ glass panes excluded: transplanting the fused
#                                        pane would paste its milky reflections back
if DETAIL_KEEP > 0 and det.sum() > 0.002 * H * Wd:
    _rl = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    _flb = cv2.cvtColor(fusedC, cv2.COLOR_BGR2LAB).astype(np.float32)
    _rL = _rl[..., 0] / 255; _fL2 = _flb[..., 0] / 255
    _ratio = (cv2.GaussianBlur(_rL, (0, 0), 12) /
              np.maximum(cv2.GaussianBlur(_fL2, (0, 0), 12), 1e-3))
    _ratio = np.clip(_ratio, 0.2, 8.0)                 # local re-exposure of the fused L
    _newL = np.clip(_fL2 * _ratio, 0, 1)               # fused texture at HDR brightness
    # SAFETY: the transplant must only add TEXTURE — never re-darken or re-brighten
    # the region as a whole. Bound its deviation from the HDR luminance to a
    # detail-sized range, so a dim scene's strong lift can never be undone here.
    _newL = _rL + np.clip(_newL - _rL, -0.22, 0.22)
    _m = cv2.GaussianBlur(det.astype(np.float32), (0, 0), 4) * DETAIL_KEEP
    _rl[..., 0] = (_rL * (1 - _m) + _newL * _m) * 255
    local_u8 = cv2.cvtColor(np.clip(_rl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"fused detail maintained on {100*det.mean():.1f}% of the image (vanity + sink/counter + lamps/fan + bulbs)")
else:
    print("no vanity/sink region found for detail-keep (or DETAIL_KEEP = 0)")

# ---- WINDOW PULL: the window shows the FUSED image's view ----------------------
# The global brightness chain unavoidably lifts the window a little too (measured
# ours L0.93 vs AutoHDR 0.85 — the view washes pale). But the fused image already
# holds the PERFECT window: highlight-recover pulled the -EV bracket into it. So
# the window's brightness is pulled back to the fused level and the view's detail
# comes back 1:1 (its colour is already the fused colour). Feathered — no frame line.
WINDOW_PULL = 0.85   #@param {type:"number"}   # 0 = off · 0.85 = default
if WINDOW_PULL > 0 and WINDOW.sum() > 0.001 * H * Wd:
    _rlw = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    _flw = cv2.cvtColor(fusedC, cv2.COLOR_BGR2LAB).astype(np.float32)
    _mw = cv2.GaussianBlur(WINDOW.astype(np.float32), (0, 0), 3) * WINDOW_PULL
    _rlw[..., 0] = _rlw[..., 0] * (1 - _mw) + _flw[..., 0] * _mw   # L from the fused window
    local_u8 = cv2.cvtColor(np.clip(_rlw, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"window pull {WINDOW_PULL}: window brightness + view restored from the fused image "
          f"on {100*WINDOW.mean():.1f}%")

# ---- TABLE mask (kept for PROTECTION only): a colour restore is no longer needed
# — the whole image already carries the fused colours (MAINTAIN_COLORS above).
# The clear-table mask is still computed because the floor / metal fixes below
# must never touch a real table.
TABLE_CLASSES = (15, 33, 44, 35)   # table, desk, chest-of-drawers, wardrobe


def _clear_only(mask, min_frac=0.006, min_solidity=0.45):
    """RESTRICTION: keep a component only if the object is detected CLEARLY —
    i.e. it is BIG enough (>= min_frac of the image) AND SOLID (a real table is a
    coherent slab; solidity = area / convex-hull area). A tiny sliver or a
    scattered/fragmented guess is DROPPED — we would rather do nothing than
    restore colour onto a mis-detection. Returns (mask, n_kept, n_dropped)."""
    out = np.zeros(mask.shape, bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    kept = dropped = 0
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        comp = (lab == i)
        if area < min_frac * mask.size:
            dropped += 1; continue
        pts = cv2.findNonZero(comp.astype(np.uint8))
        hull_area = cv2.contourArea(cv2.convexHull(pts)) if pts is not None else 0
        solidity = area / max(hull_area, 1.0)
        if solidity >= min_solidity:
            out |= comp; kept += 1
        else:
            dropped += 1
    return out, kept, dropped


wood = np.isin(seg, TABLE_CLASSES) & ~WINDOW
wood, _k, _d = _clear_only(wood)                    # only a CLEARLY-shown table qualifies
if _d:
    print(f"table: {_k} clear table(s) kept, {_d} unclear/partial detection(s) skipped")
# (the old RESTORE_COLOR / MATERIAL_RESTORE blocks are gone — MAINTAIN_COLORS above
#  already keeps every material's fused colour, gate-free and over the whole image.)

# ---- FLOOR COLOUR: ONE parameter that sets the floor's colour directly ---------
# (replaces the old FLOOR_RESTORE + FLOOR_UNIFORM pair, which fought each other)
# Pick the tone; every floor pixel's COLOUR is blended to that ONE even tone
# (like AutoHDR's perfectly uniform floor). L is never touched, so the planks,
# grain and texture stay exactly as they are — only the colour is set.
#   auto      = the floor's own natural tone (median of the FUSED floor, clamped
#               to a realistic wood range) — right for most scenes
#   warm oak  = a+2  b+9   (the AutoHDR-measured floor tone)
#   light oak = a+1.5 b+6
#   golden    = a+3  b+12
#   grey      = a+0  b+1   (grey/washed floors stay grey)
#   keep      = off (floor left exactly as the chain made it)
FLOOR_COLOR = "auto"   #@param ["auto", "warm oak", "light oak", "golden", "grey", "keep"]
floor = np.isin(seg, (3, 13)) & ~WINDOW          # floor, ground (rug 28 left out on purpose)
_PRESETS = {"warm oak": (2.0, 9.0), "light oak": (1.5, 6.0),
            "golden": (3.0, 12.0), "grey": (0.0, 1.0)}
if FLOOR_COLOR != "keep" and floor.sum() > 0.002 * H * Wd:
    if FLOOR_COLOR == "auto":
        f_lab = cv2.cvtColor(fusedC, cv2.COLOR_BGR2LAB).astype(np.float32)   # cast-corrected fused
        tgt_a = float(np.clip(np.median((f_lab[..., 1] - 128)[floor]), 0.0, 4.0))
        tgt_b = float(np.clip(np.median((f_lab[..., 2] - 128)[floor]), 4.0, 10.0))
        # in a WHITE scene the floor's warmth is lighting cast, not wood — follow
        # the scene decision (AutoHDR's white-kitchen floor measures b+1)
        tgt_a = tgt_a * (1 - white_scene)
        tgt_b = tgt_b * (1 - white_scene) + 1.0 * white_scene
        # the LLM saw the floor's true material/tone — its verdict wins in auto mode
        _lfl = (LLM or {}).get("floor") or {}
        _LT = {"warm_oak": (2.0, 9.0), "light_oak": (1.5, 6.0),
               "golden": (3.0, 12.0), "grey": (0.0, 1.0)}
        if _lfl.get("tone") == "grey":                    # explicitly grey -> grey
            tgt_a, tgt_b = _LT["grey"]
        elif _lfl.get("material") == "wood" and _lfl.get("tone") in _LT and white_scene < 0.5:
            tgt_a, tgt_b = _LT[_lfl["tone"]]
        # any other tile/stone tone: keep the measured auto tone (a warm marble
        # floor stays warm — AutoHDR keeps the bathroom hex floor at b+12)
    else:
        tgt_a, tgt_b = _PRESETS[FLOOR_COLOR]
    lab = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    fa = lab[..., 1] - 128; fb = lab[..., 2] - 128
    m = cv2.GaussianBlur(floor.astype(np.float32), (0, 0), 6) * 0.85   # fixed, fully even
    lab[..., 1] = (fa * (1 - m) + tgt_a * m) + 128
    lab[..., 2] = (fb * (1 - m) + tgt_b * m) + 128
    local_u8 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"floor colour set to '{FLOOR_COLOR}' (a{tgt_a:+.1f} b{tgt_b:+.1f}) — one even tone, texture kept")

# ---- make the FLOOR read CLEARER: a mild local-contrast (clarity) boost on the
# floor only, so the planks / wood grain / joins are defined instead of looking
# like a flat wash. Wide-radius and masked to the floor, feathered — it lifts
# texture, not edges, so it cannot ring or halo. 0 = off.
FLOOR_CLARITY = 0.25   #@param {type:"number"}   # 0 = off, 0.25 = default, 0.5 = strong
if FLOOR_CLARITY > 0 and floor.sum() > 0.002 * H * Wd:
    _f = local_u8.astype(np.float32) / 255
    _wide = cv2.GaussianBlur(_f, (0, 0), 8)                       # texture-scale, not edges
    _m = cv2.GaussianBlur(floor.astype(np.float32), (0, 0), 6)[..., None] * FLOOR_CLARITY
    local_u8 = (np.clip(_f + _m * (_f - _wide), 0, 1) * 255).astype("uint8")
    print(f"floor clarity +{FLOOR_CLARITY} on {100*floor.mean():.1f}% (plank/grain definition)")

# ---- WALLS -> CLEAN WHITE · TILES / MARBLE -> natural tone (texture-aware) -----
# The PAINTED wall is pushed to clean white (AutoHDR: upper bathroom walls b-1),
# but any TILED / MARBLE part of the same wall class (wainscot, shower walls,
# backsplash) is recognised by its TEXTURE in the fused image — grout lines and
# veining — and is only SOFTENED toward a natural neutral tone (AutoHDR keeps
# shower tiles at b+12), never forced white. The veining itself never moves, and
# a COOL painted colour (a green/blue feature wall) is never touched — only warm
# cast/paint is whitened. Measured: paint tex 0.012-0.021 vs tiles 0.040-0.063.
NEUTRALIZE_WALLS = 0.95  #@param {type:"number"}   # 0 = off · 0.95 = paint FULLY white, tiles natural
TILE_SOFTEN = 0.45       #@param {type:"number"}   # how much of the whitening TILES get
                                                   # (0 = tiles fully kept · 1 = tiles like paint)
wc = np.isin(seg, (0, 5, 1))
wc = cv2.dilate(wc.astype(np.uint8), np.ones((max(3, int(Wd * 0.008) | 1),) * 2, np.uint8)).astype(bool)
wc = wc | WHITEN_EXTRA                             # LLM-judged white cabinets: colour whitened
wc = wc & ~WINDOW & ~wood & ~floor                 # never grey the window / table / floor
if NEUTRALIZE_WALLS > 0 and wc.sum() > 0.002 * H * Wd:
    _fl5 = cv2.cvtColor(fusedC, cv2.COLOR_BGR2LAB).astype(np.float32)
    _L5 = _fl5[..., 0] / 255; _a5 = _fl5[..., 1] - 128; _b5 = _fl5[..., 2] - 128
    # TILE-NESS: how much grout/vein structure the fused surface carries around
    # each pixel — smooth paint reads ~0, tiled/marble areas read high.
    _hpL = np.abs(_L5 - cv2.GaussianBlur(_L5, (0, 0), 3))
    _hpB = np.abs(_b5 - cv2.GaussianBlur(_b5, (0, 0), 3)) / 30.0
    _tex = cv2.GaussianBlur(_hpL + _hpB, (0, 0), max(10, int(Wd * 0.015)))
    tile = np.clip((_tex - 0.026) / 0.012, 0, 1)   # 0 = smooth paint · 1 = tile/marble
    # a single strong EDGE (door frame, trim, wall corner) is not a tile — tiles
    # are AREAL texture. Morphological opening removes thin edge-lines from the
    # tile map, so the wall right next to a door frame is whitened fully and
    # EVENLY instead of keeping an orange strip along the frame.
    _tk = (max(5, int(Wd * 0.01)) | 1)
    tile = cv2.morphologyEx((tile > 0.5).astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((_tk, _tk), np.uint8)).astype(np.float32) * tile
    # tiles are DENSE texture (grout/veins everywhere — measured density 0.24-0.79),
    # while a lamp chain / plant stems / a frame edge crossing the wall is SPARSE
    # texture (measured 0.003). Sparse edges never make a tile — so the paint
    # BEHIND objects still gets FULL, even whitening (no orange island around a
    # pendant lamp or an orchid in front of the wall).
    _edge5 = ((_hpL + _hpB) > 0.05).astype(np.float32)
    _dens5 = cv2.GaussianBlur(_edge5, (0, 0), max(10, int(Wd * 0.015)))
    tile = tile * np.clip((_dens5 - 0.08) / 0.08, 0, 1)
    # VEIN PROTECTION: veining only exists ON tiles/marble — on smooth paint the
    # same test would spare warm paint-texture speckle and leave orange spots,
    # so the vein map is gated by the tile map (paint gets a perfectly even white).
    vein = np.clip((_b5 - cv2.GaussianBlur(_b5, (0, 0), 25) - 3) / 5, 0, 1)
    vein = vein * np.clip(tile * 2, 0, 1)
    # COOL-PAINT PROTECTION: a genuinely green/blue wall is real colour, not cast.
    # Sharp ramp — a near-neutral wall gets FULL, even whitening (no patchiness).
    _cool = np.clip((_b5 + 3) / 2, 0, 1) * np.clip((_a5 + 6) / 3, 0, 1)
    strength = NEUTRALIZE_WALLS * (1 - (1 - TILE_SOFTEN) * tile) * _cool
    lab = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = cv2.GaussianBlur(wc.astype(np.float32), (0, 0), 4) * strength * (1 - vein)
    lab[..., 1] = lab[..., 1] * (1 - m) + 128 * m   # a -> neutral
    lab[..., 2] = lab[..., 2] * (1 - m) + 128 * m   # b -> neutral
    local_u8 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"walls whitened on {100*float((wc & (tile < 0.3)).mean()):.1f}% (paint) · "
          f"tiles kept natural on {100*float((wc & (tile > 0.5)).mean()):.1f}% · "
          f"veins spared on {100*float((wc & (vein > 0.5)).mean()):.1f}%")
else:
    print("walls untouched (NEUTRALIZE_WALLS = 0)")

# ---- WALL SHADE EVENING (no-tile walls): a painted wall must read ONE even
# bright white. The colour whitening removes the TINT but not the DARKNESS —
# warm SHADE BANDS (the strip above a cabinet soffit, the tan band beside a
# door) stay as darker patches because the wall exposure targeting skips pixels
# far from the wall median. Here, ONLY when the walls carry no real tiles or
# marble, every paint-like shade band is lifted (low-frequency) to the wall's
# own level and its leftover warm tint cleaned to white. Paint-like = smooth +
# low chroma + warm-ish — so a cabinet-labelled SOFFIT that is really paint is
# caught too, while real wood (high chroma), a cool/green wall (a/b gates), the
# curtains, window, floor and furniture never move. Feathered + low-freq -> no halos.
WALL_EVEN = 0.9    #@param {type:"number"}   # 0 = off · 0.9 = default
_tile_frac = float((_tex_v[wc & ~WINDOW] > 0.026).mean()) if (wc & ~WINDOW).sum() > 1000 else 1.0
if WALL_EVEN > 0 and _tile_frac < 0.15 and wc.sum() > 0.002 * H * Wd:
    # THREE PASSES: one feathered pass always leaves ~15-20% of the deficit
    # behind (measured: bands closed 0.07 -> 0.03, still faintly visible). Each
    # extra pass measures the REMAINING deficit and removes it — two passes
    # reached 0.82 on a 0.85 wall (still a visible band beside the door), the
    # third converges to within ~0.01.
    for _pass_e in range(3):
        _le = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        _Le = _le[..., 0] / 255
        _ae = _le[..., 1] - 128; _be = _le[..., 2] - 128
        _che = np.sqrt(_ae * _ae + _be * _be)
        # MUD-vs-COLOUR cap: tungsten mud on a shaded wall return is YELLOW-only
        # (a ~ 0, b up to +20, chroma ~20) — a fixed chroma<16/18 cap protected
        # exactly those muddiest patches ("too warm to clean"). Real warm
        # material (oak, tan fabric) always carries red too (a >= +5). So the
        # cap is a-gated: pure yellow may clean up to chroma 26, anything with
        # real red keeps the old strict cap.
        _cap_e = 18 + 8 * np.clip((5 - _ae) / 5, 0, 1)
        # texture bound 0.020 -> 0.026 (still under the 0.026 tile threshold):
        # a lightly-textured STUCCO wall return is paint too — before, the mud
        # on it failed _paintlike and was never cleaned when OneFormer labelled
        # the return 'door' instead of 'wall'.
        _paintlike = (_tex_v < 0.026) & (_che < _cap_e) & (_Le > 0.40) & (_Le < 0.92)
        # CONTACT-SHADE KEEP: the wall right around a DARK object (console, plant,
        # decor) carries the object's natural contact shade — evening it flat
        # makes the object float in a white GLOW (the "halo" look). A thin ring
        # around every dark non-wall object is left out of the evening; open-wall
        # shade bands (a door-side strip, a soffit band) are far from any dark
        # object and still converge fully.
        _darkobj = (_Le < 0.50) & ~np.isin(seg, (0, 1, 5))
        _objring = (cv2.dilate(_darkobj.astype(np.uint8),
                               np.ones((max(9, int(Wd * 0.012) | 1),) * 2, np.uint8)).astype(bool)
                    & ~_darkobj)
        # wc is texture-gated too: a TILED backsplash is wall-CLASS but never
        # paint — evening it bleached the tiles to stark white bloom. And real
        # furniture surfaces (chair/sofa leather 19/23/30, the counter 45) must
        # never turn "paintlike" just because the de-warm steps dropped their
        # chroma — they are excluded by class like the rest of the furniture.
        _evenm = (((wc & (_tex_v < 0.026)) | _paintlike) & ~WINDOW & ~floor & ~wood
                  & ~_objring
                  & ~np.isin(seg, (15, 18, 19, 23, 30, 33, 36, 45, 82, 85, 139, 200,
                                   71, 50, 118, 124, 129, 107, 133)))
        _wref = float(np.median(_Le[wc & ~WINDOW])) if (wc & ~WINDOW).sum() > 1000 else 0.80
        # MASK-NORMALISED local level: the wall's local brightness is estimated
        # from WALL PIXELS ONLY — blur(L*mask)/blur(mask). A curtain, console or
        # chair contributes NOTHING to the reference, so the wall around a dark
        # object can never be over-lifted: the halo is mathematically impossible
        # (a plain Gaussian mixed objects in and created glow rings around them).
        _mf_e = _evenm.astype(np.float32)
        _sig_e = max(15, int(Wd * 0.015))
        _den_e = cv2.GaussianBlur(_mf_e, (0, 0), _sig_e)
        _lo_e = cv2.GaussianBlur(_Le * _mf_e, (0, 0), _sig_e) / np.maximum(_den_e, 1e-4)
        _lo_e = np.where(_den_e > 0.10, _lo_e, _wref)      # far from any wall -> no deficit
        _deficit = np.clip(_wref - _lo_e, 0, 0.22)         # how far below the wall level
        # LIFT gate: low-chroma shade of ANY tint — after the colour whitening the
        # bands are NEUTRAL grey (measured b+0), so a warm-only gate would cut the
        # lift to a third exactly where it is needed. Blue/green still protected.
        _gate_L = (np.clip((_cap_e - _che) / 6, 0, 1)      # paint + yellow mud — real colour never lifted
                   * np.clip((_be + 2) / 2, 0, 1)          # blue shade stays (window light)
                   * np.clip((_ae + 2) / 2, 0, 1))         # green (sage curtains/walls) safe
        _m_e = cv2.GaussianBlur(_evenm.astype(np.float32), (0, 0), 6) * WALL_EVEN * _gate_L
        _le[..., 0] = np.clip(_Le + _deficit * _m_e, 0, 1) * 255
        _warm_e = np.clip((_be - 2) / 4, 0, 1)             # leftover warm tint -> clean white
        _le[..., 1] = _ae * (1 - _m_e * _warm_e) + 128
        _le[..., 2] = _be * (1 - _m_e * _warm_e) + 128
        local_u8 = cv2.cvtColor(np.clip(_le, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"wall shade evening (3 passes): no tiles on the walls ({100*_tile_frac:.0f}% tile "
          f"texture) -> shade bands converged to the wall level L{_wref:.2f} + tint cleaned")
elif WALL_EVEN > 0:
    print(f"wall shade evening skipped — walls carry tile/marble texture "
          f"({100*_tile_frac:.0f}%), kept natural (user rule)")

# ---- CLEAN the BULB GLOW: neutralise the warm reflection halo around each light
# so the ceiling/wall around a bulb stays clean white (the bulb core itself is left
# alone — it keeps its natural glow). The GLOW mask was found above; here we just
# pull its a,b toward neutral, feathered so there is no ring/edge.
NEUTRALIZE_GLOW = 0.85   #@param {type:"number"}   # 0 = off, 0.85 = default
# Runs in EVERY scene now: since the emitter test keeps only REAL bulbs in the
# BULB mask, the glow ring lands only around actual lights — and its chroma gate
# (< 18) keeps real coloured materials out of it. Measured on the wood kitchen:
# the pendant's warm halo on the wall was b+12 vs AutoHDR's b+2.
_glow_amt = NEUTRALIZE_GLOW
if _glow_amt > 0 and GLOW.sum() > 0.0005 * H * Wd:
    lab = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = cv2.GaussianBlur(GLOW.astype(np.float32), (0, 0), 6) * _glow_amt
    lab[..., 1] = lab[..., 1] * (1 - m) + 128 * m
    lab[..., 2] = lab[..., 2] * (1 - m) + 128 * m
    local_u8 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"neutralised bulb-glow halo on {100*GLOW.mean():.2f}% (clean ceiling around lights)")

# ---- BULBS -> WHITE: the lit bulbs/lamps themselves read clean WHITE instead of
# warm yellow — only their COLOUR is neutralised; the brightness and the shape of
# the glow stay exactly as they are (the light still looks lit, just white).
BULB_WHITE = 0.9   #@param {type:"number"}   # 0 = keep natural warm bulbs · 0.9 = white (default)
if BULB_WHITE > 0 and BULB.any():
    lab = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    # include the bright CORONA right around each bulb — that halo is what the eye
    # reads as "the bulb's colour". Anything that bright, that close to a confirmed
    # emitter, is light (chroma cap keeps a genuinely coloured shade safe).
    _bw = cv2.dilate(BULB.astype(np.uint8),
                     np.ones((max(5, int(Wd * 0.012)) | 1,) * 2, np.uint8)).astype(bool)
    _bL = lab[..., 0] / 255
    _bch = np.sqrt((lab[..., 1] - 128) ** 2 + (lab[..., 2] - 128) ** 2)
    # + ON the lamp/fan labels: every LIT part goes white, even dim diffused glass
    # (smoky lantern, frosted dome). Lit = bright-ish now AND still faintly glowing
    # in the darkest bracket (an unlit shade reflects almost nothing there); the
    # chroma cap keeps a genuinely coloured glass shade safe.
    _bw = BULB | (_bw & (_bL > 0.70) & (_bch < 35)) \
               | (np.isin(seg, (36, 82, 85, 139)) & (_bL > 0.60) & (_bch < 35) & (_dklu > 0.22))
    m = cv2.GaussianBlur(_bw.astype(np.float32), (0, 0), 3) * BULB_WHITE
    lab[..., 1] = lab[..., 1] * (1 - m) + 128 * m
    lab[..., 2] = lab[..., 2] * (1 - m) + 128 * m
    local_u8 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"bulbs whitened on {100*BULB.mean():.2f}% (lamps read clean white, brightness kept)")

# ---- NEUTRAL METALS: stainless steel / chrome appliances must stay SILVER ------
# Steel reflects the room's warm lighting, so the SOURCE image already carries a
# yellow tint on the oven/stove/fridge — no warmth setting can remove that, it has
# to be actively neutralised. The appliance classes are segmented and their colour
# (LAB a,b) pulled to neutral; a chroma gate (<16) makes sure anything genuinely
# coloured inside the box (a red kettle on the stove) is untouched, and L is never
# changed so the brushed-metal texture and reflections stay exactly as they are.
# Measured: microwave b+4.0 -> ~+1 (AutoHDR +1.5), dishwasher b+2.7 -> ~+0.8.
NEUTRAL_METALS = 0.75   #@param {type:"number"}   # 0 = off, 0.75 = default, 1 = fully silver
METAL_CLASSES = (71, 50, 118, 124, 129, 107, 133)   # stove, refrigerator, oven,
                                                    # microwave, dishwasher, washer, hood
metal = np.isin(seg, METAL_CLASSES) & ~WINDOW
# COMPLETE each appliance: OneFormer often labels only the COOKTOP as 'stove' and
# misses the oven door / drawer below it (the "upper oven perfect, bottom one not"
# bug). Extend every detected component through its bounding box and DOWNWARD by
# 80% of its height, so the whole appliance body is covered; the chroma gate and
# the floor/table exclusions below keep everything else safe.
if metal.any():
    _n, _, _st, _ = cv2.connectedComponentsWithStats(metal.astype(np.uint8))
    _box = np.zeros((H, Wd), bool)
    for _i in range(1, _n):
        if _st[_i, cv2.CC_STAT_AREA] < 0.0005 * H * Wd:
            continue
        _x, _y = _st[_i, cv2.CC_STAT_LEFT], _st[_i, cv2.CC_STAT_TOP]
        _w2, _h2 = _st[_i, cv2.CC_STAT_WIDTH], _st[_i, cv2.CC_STAT_HEIGHT]
        _pad = max(2, int(_w2 * 0.03))
        # extend DOWN by 2.5x the detected height OR 1.1x the width (whichever is
        # larger) — a range/oven front is tall, and the label often covers only
        # the small cooktop strip at the top
        _down = int(max(_h2 * 2.5, _w2 * 1.1))
        _box[max(0, _y - _pad):min(H, _y + _down),
             max(0, _x - _pad):min(Wd, _x + _w2 + _pad)] = True
    # HARD exclusions: the completed box may spill onto the WOOD CABINETS around
    # an appliance (a microwave between cabinets!) — cabinet/shelf classes are
    # never touched, along with window/floor/table.
    metal = (metal | _box) & ~WINDOW & ~floor & ~wood & ~np.isin(seg, (10, 24, 44, 35))
if NEUTRAL_METALS > 0 and metal.sum() > 0.001 * H * Wd:
    lab = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    ma = lab[..., 1] - 128; mb = lab[..., 2] - 128
    mch = np.sqrt(ma * ma + mb * mb)
    gate = np.clip((24 - mch) / 6, 0, 1)             # full neutralize up to ch~18 (covers even
                                                     # strongly tinted steel) but ZERO by ch 24 —
                                                     # wood (ch 25+) can never be greyed, even
                                                     # where the box overlaps it
    m = cv2.GaussianBlur(metal.astype(np.float32), (0, 0), 3) * NEUTRAL_METALS * gate
    lab[..., 1] = ma * (1 - m) + 128
    lab[..., 2] = mb * (1 - m) + 128
    local_u8 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"neutralised metal appliances on {100*metal.mean():.1f}% (full body incl. oven door, stainless stays silver)")

# ---- SUN-TONE: tone down the big sunlit near-white patches ---------------------
# Where direct sunlight lands (walls / floor near a window or door) the result
# goes flat paper-white while AutoHDR keeps those areas visibly TONED (~0.05
# darker — measured: far-door 0.84, sunlit wall 0.79 vs ours 0.86-0.89). A soft
# knee above 0.84 compresses them toward that level. The CEILING class and the
# BULBS are excluded (both are supposed to stay bright); monotonic -> no halos.
SUN_TONE = 0.8   #@param {type:"number"}   # 0 = off, 0.8 = default, 1 = strongest
if SUN_TONE > 0:
    _f = local_u8.astype(np.float32) / 255
    _lu = _f.mean(2)
    # STRUCTURE-PRESERVING: the knee is applied to the WIDE-BLURRED luminance and
    # only the resulting smooth ratio scales the pixels. The sunlit patch is toned
    # DOWN as a whole, but its internal light gradient and detail stay 1:1 — the
    # old per-pixel knee was compressing every bright wall's texture into one flat
    # milky tone. The smooth ratio cannot halo; ceiling, bulbs AND the window view
    # are excluded (the view must keep its own brightness).
    _lo = cv2.GaussianBlur(_lu, (0, 0), max(15, int(Wd * 0.012)))
    _knee = 0.87            # raised from 0.84: an ordinary BRIGHT WHITE wall must not
                            # be pulled back down — only genuine sun patches above it
    _slope = 1 - 0.45 * SUN_TONE
    _new = np.where(_lo > _knee, _knee + (_lo - _knee) * _slope, _lo)
    _ratio = np.where(_lo > 1e-6, _new / np.maximum(_lo, 1e-6), 1.0)
    _keep = cv2.GaussianBlur(((seg == 5) | BULB | WINDOW).astype(np.float32), (0, 0), 5)
    _ratio = 1.0 + (_ratio - 1.0) * (1.0 - _keep)
    local_u8 = (np.clip(_f * _ratio[..., None], 0, 1) * 255).astype("uint8")
    print(f"sun-tone {SUN_TONE}: sunlit patches toned as a whole — light gradient kept (ceiling/bulbs/window untouched)")

# ---- HIGHLIGHT-GRADIENT RESTORE: every gamma-brightening step mathematically
# compresses the TOP of the tonal range, so the natural falloff of light across a
# bright wall flattens into one milky tone (measured: bright wall spread 0.009 vs
# AutoHDR 0.043 — while the fused image itself still holds 0.044). This REPLACES
# the flattened structure in the bright zones with the fused image's own light
# structure at the new brightness: it is the image's own structure (not an edge
# amplification), so it cannot ring or halo; brightness and colour stay exactly
# as set above. The window view and the vanity/sink (already detail-kept) are out.
HIGHLIGHT_STRUCTURE = 0.8   #@param {type:"number"}   # 0 = off · 0.8 = AutoHDR-level (default)
if HIGHLIGHT_STRUCTURE > 0:
    _rl9 = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    _rL9 = _rl9[..., 0] / 255
    _fl9 = cv2.cvtColor(fusedC, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 0] / 255
    _sig9 = max(40, int(Wd * 0.05))                    # wide: the light FALLOFF scale
    _fst = _fl9 - cv2.GaussianBlur(_fl9, (0, 0), _sig9)   # fused light structure
    _rst = _rL9 - cv2.GaussianBlur(_rL9, (0, 0), _sig9)   # what is left of it now
    _delta = np.clip(_fst - _rst, -0.2, 0.2)               # the MISSING structure
    # fade the restore out as a pixel approaches white, so it re-creates the
    # gradient in the 0.78-0.93 zone (where the flattening happened) without
    # pushing the already-near-white pixels over into blown territory
    _delta = _delta * (1 - np.clip((_rL9 - 0.93) / 0.05, 0, 1))
    _wgt9 = np.clip((cv2.GaussianBlur(_rL9, (0, 0), 20) - 0.78) / 0.10, 0, 1)
    _wgt9 = _wgt9 * (1 - cv2.GaussianBlur((WINDOW | det).astype(np.float32), (0, 0), 5))
    _rl9[..., 0] = np.clip(_rL9 + _delta * _wgt9 * HIGHLIGHT_STRUCTURE, 0, 1) * 255
    local_u8 = cv2.cvtColor(np.clip(_rl9, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"highlight structure restored on {100*float((_wgt9 > 0.5).mean()):.1f}% "
          f"(bright walls keep their natural light falloff)")

# ---- DEPTH KEEP: don't over-lift the genuinely DARK zones ----------------------
# The shadow-fill + analysis dodge lift dark zones so nothing is ever crushed —
# but on BIG dark areas they overshoot and the image loses its depth: measured
# the soffit shadow above the cabinets at 0.50 vs AutoHDR's 0.28 and a dark
# hallway at 0.64 vs 0.47 — those areas read flat and hazy. This caps the
# LOW-frequency lift of genuinely-dark fused zones at +0.20 stops of tone: the
# texture is untouched, a strong lift is still allowed (nothing gets crushed),
# only the overshoot beyond the cap is taken back. Wide-blurred -> cannot halo.
DEPTH_KEEP = 0.8   #@param {type:"number"}   # 0 = off · 0.8 = default
if DEPTH_KEEP > 0:
    _rlD = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    _rLD = _rlD[..., 0] / 255
    _fLD = cv2.cvtColor(fusedC, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 0] / 255
    _sigD = max(20, int(Wd * 0.02))
    _rloD = cv2.GaussianBlur(_rLD, (0, 0), _sigD)
    _floD = cv2.GaussianBlur(_fLD, (0, 0), _sigD)
    _darkm = np.clip((0.55 - _floD) / 0.20, 0, 1)          # genuinely dark fused zones
    _darkm = _darkm * (1 - cv2.GaussianBlur(WINDOW.astype(np.float32), (0, 0), 5))
    _excess = np.maximum(_rloD - _floD - 0.20, 0)          # lift beyond +0.20 = overshoot
    _rlD[..., 0] = np.clip(_rLD - _excess * _darkm * DEPTH_KEEP, 0, 1) * 255
    local_u8 = cv2.cvtColor(np.clip(_rlD, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"depth keep: over-lift taken back on {100*float((_excess * _darkm > 0.02).mean()):.1f}% "
          f"(dark zones keep their depth — nothing crushed)")

# ================= LABEL-INDEPENDENT FINAL GUARDS =============================
# The three residual defects (milky pane blob, door-side mud, washed sun
# reflections) all survived because every earlier pass is gated by the OneFormer
# CLASS — and OneFormer labels the pane 'wall', the mud 'floor', run-dependent.
# These three guards work from the IMAGE itself, so no label flip can dodge them.

# ---- 1) CABINET-GLASS RESTORE: a dark region ENCLOSED BY WOOD is a glass pane
# (or a TV screen) — whatever the chain did to it (wall lift, whitening, shade
# evening, vanity brightening — its label is unreliable), it is restored to its
# FUSED appearance: dark, warm glass with only its real reflections.
_fgl = cv2.cvtColor(fusedC, cv2.COLOR_BGR2LAB).astype(np.float32)
_fgL = _fgl[..., 0] / 255
_fga = _fgl[..., 1] - 128; _fgb = _fgl[..., 2] - 128


def _fill_holes(m):
    """A milky reflection blob INSIDE a dark pane is a hole of the dark
    component — it must be restored WITH the pane, so holes are filled."""
    ff = m.astype(np.uint8).copy()
    mm = np.zeros((H + 2, Wd + 2), np.uint8)
    cv2.floodFill(ff, mm, (0, 0), 1)
    return m | (ff == 0)


_gcand = (_fgL < 0.50) & ~WINDOW & ~BULB
_gcand = cv2.morphologyEx(_gcand.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((5, 5), np.uint8)).astype(bool)
_woodish = (_fgb > 16) & (_fga > 4) & (_fgL > 0.22) & (_fgL < 0.80)
GLASS_DARK = np.zeros((H, Wd), bool)
_ng, _lg, _sg, _ = cv2.connectedComponentsWithStats(_gcand.astype(np.uint8))
_rkg = max(9, int(Wd * 0.012) | 1)
for _j in range(1, _ng):
    if not (0.0002 * H * Wd < _sg[_j, cv2.CC_STAT_AREA] < 0.03 * H * Wd):
        continue
    _cmg = _fill_holes(_lg == _j)
    _ring = cv2.dilate(_cmg.astype(np.uint8), np.ones((_rkg, _rkg), np.uint8)).astype(bool) & ~_cmg
    if _ring.sum() and float(_woodish[_ring].mean()) > 0.55:
        GLASS_DARK |= _cmg
if GLASS_DARK.any():
    _mg = cv2.GaussianBlur(GLASS_DARK.astype(np.float32), (0, 0), 3) * 0.85
    _rgl = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    # gentle exposure match: the pane may sit a LITTLE above its fused level
    # (the whole room brightened) but never more than +0.10
    _tgtL = np.minimum(_rgl[..., 0] / 255, _fgL + 0.10)
    _rgl[..., 0] = (_rgl[..., 0] / 255 * (1 - _mg) + _tgtL * _mg) * 255
    _rgl[..., 1] = _rgl[..., 1] * (1 - _mg) + (_fga + 128) * _mg
    _rgl[..., 2] = _rgl[..., 2] * (1 - _mg) + (_fgb + 128) * _mg
    local_u8 = cv2.cvtColor(np.clip(_rgl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"cabinet-glass restore: {100*GLASS_DARK.mean():.2f}% wood-enclosed glass "
          f"kept dark (fused look — no milky blobs, label-independent)")
else:
    print("cabinet-glass restore: no wood-enclosed glass found")

# ---- 2) MUD CLEANER: a SMALL warm patch floating in bright NEUTRAL
# surroundings is dirt (tungsten mud on a whitened wall / door return), not
# material. Cleaned by its own profile — the class gates of the evening pass
# are what kept missing it (OneFormer calls the spot 'floor' or 'door').
_ml = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
_mL = _ml[..., 0] / 255; _ma = _ml[..., 1] - 128; _mb = _ml[..., 2] - 128
_mudc = ((_mb > 8) & (_mb < 24) & (_ma > -3) & (_ma < 6)
         & (_mL > 0.45) & (_mL < 0.92) & ~WINDOW & ~BULB & ~wood & ~GLASS_DARK
         & ~np.isin(seg, (18, 36, 82, 85, 139, 200)))
MUD = np.zeros((H, Wd), bool)
_nm2, _lm2, _sm2, _ = cv2.connectedComponentsWithStats(_mudc.astype(np.uint8))
_rkm = max(9, int(Wd * 0.010) | 1)
for _j in range(1, _nm2):
    if _sm2[_j, cv2.CC_STAT_AREA] > 0.015 * H * Wd:
        continue                                   # a big warm AREA is a material
    _cmm = _lm2 == _j
    _rgm = (cv2.dilate(_cmm.astype(np.uint8), np.ones((_rkm, _rkm), np.uint8)).astype(bool)
            & ~_mudc)
    if _rgm.sum() < 20:
        continue
    _rch = np.sqrt(_ma[_rgm] ** 2 + _mb[_rgm] ** 2)
    if float(np.median(_rch)) < 9 and float(np.median(_mL[_rgm])) > 0.72:
        MUD |= _cmm                                # warm speck in a white world = dirt
if MUD.any():
    _mm = cv2.GaussianBlur(MUD.astype(np.float32), (0, 0), 4) * 0.9
    _nmud = (~_mudc).astype(np.float32)
    _refm = (cv2.GaussianBlur(_mL * _nmud, (0, 0), 15)
             / np.maximum(cv2.GaussianBlur(_nmud, (0, 0), 15), 1e-4))
    _ml[..., 0] = np.clip(_mL + np.clip(_refm - _mL, 0, 0.20) * _mm, 0, 1) * 255
    _ml[..., 1] = _ma * (1 - _mm) + 128
    _ml[..., 2] = _mb * (1 - _mm) + 128
    local_u8 = cv2.cvtColor(np.clip(_ml, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"mud cleaner: {100*MUD.mean():.2f}% warm patches in white surroundings "
          f"neutralised + lifted (label-independent)")
else:
    print("mud cleaner: no mud patches found")

# ---- 3) SUN KEEP: sunlight reflections (the floor sun path, counter glare,
# lit chair tops) stay as the FUSED image shows them — their warm tint comes
# back and they may only rise as much as their OWN surface rose, so the chain
# can never wash them pale-blue or push them toward clipping.
# label-independent surface add-on: any TEXTURED surface in the fused image
# (granite, tile backsplash, wood grain — _tex_v > 0.026) is MATERIAL, never
# open wall paint — its reflections/bevel highlights get the sun-keep even when
# OneFormer labels it 'wall' (that mislabel is what let the counter glare and
# the tile bevels bloom to white).
_sk_surf = ((floor | np.isin(seg, (45, 15, 19, 23, 30, 33, 10)) | VANITY | wood
             | (_tex_v > 0.026))
            & ~WINDOW & ~BULB & ~np.isin(seg, (8, 14, 18)))
# a reflection is RELATIVE: the fused sun pool on a dark floor sits at L~0.55 —
# far below any absolute "bright" gate — but clearly ABOVE its own surface's
# local level. Mask-normalised local level over the surfaces, wide radius.
# The reference here is the PURE fused image (not fusedC): the cast correction
# over-cools bright cool pixels, and restoring toward it turned the sun pool
# BLUER — the pure fused holds the reflection's true daylight tint.
_pfl = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB).astype(np.float32)
_pfL = _pfl[..., 0] / 255
_pfa = _pfl[..., 1] - 128; _pfb = _pfl[..., 2] - 128
_sfm = _sk_surf.astype(np.float32)
_sigl = max(60, int(Wd * 0.045))
_locL = (cv2.GaussianBlur(_pfL * _sfm, (0, 0), _sigl)
         / np.maximum(cv2.GaussianBlur(_sfm, (0, 0), _sigl), 1e-4))
_sun = _sk_surf & ((_pfL > _locL + 0.10) | (_pfL > 0.72))
if _sun.sum() > 0.0005 * H * Wd:
    _sl2 = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    _sL = _sl2[..., 0] / 255
    _bm = (_sk_surf & ~_sun).astype(np.float32)   # the surface around the reflection
    _sigs = max(20, int(Wd * 0.02))
    _rise = (cv2.GaussianBlur((_sL - _pfL) * _bm, (0, 0), _sigs)
             / np.maximum(cv2.GaussianBlur(_bm, (0, 0), _sigs), 1e-4))
    _rise = np.clip(_rise, 0.0, 0.25)
    _caps = np.minimum(_sL, _pfL + _rise)         # no extra enhancement beyond the surface's own lift
    _ms = cv2.GaussianBlur(_sun.astype(np.float32), (0, 0), 5) * 0.8
    _sl2[..., 0] = (_sL * (1 - _ms) + _caps * _ms) * 255
    _sl2[..., 1] = _sl2[..., 1] * (1 - _ms) + (_pfa + 128) * _ms
    _sl2[..., 2] = _sl2[..., 2] * (1 - _ms) + (_pfb + 128) * _ms
    local_u8 = cv2.cvtColor(np.clip(_sl2, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"sun keep: {100*_sun.mean():.1f}% sun/glare reflections kept at their "
          f"fused warmth + level (not enhanced)")
else:
    print("sun keep: no sun reflections found on the surfaces")

# the ONE anti-clip pass of this cell (knee 0.93, firmer slope): only the very
# top of the range is compressed so nothing clips to paper white, while bright
# walls keep their natural glow and gradient — AutoHDR lets ~11% of a bright
# wall sit above 0.93, and the old stacked compressions were erasing exactly that.
local_u8 = (_soft_highlights(local_u8.astype(np.float32) / 255, knee=0.93, slope=0.5) * 255).clip(0, 255).astype("uint8")

cv2.imwrite("hdr_result_local.jpg", local_u8, [cv2.IMWRITE_JPEG_QUALITY, 95])

# histograms (numbers)
show_hist(result, "GLOBAL result (CELL 4)")
show_hist(local_u8, "FUSED-BASED result (CELL 5)")

# GLOBAL (CELL 4) vs FUSED-BASED (CELL 5) side by side so you can compare the two
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
ax[0].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); ax[0].axis("off")
ax[0].set_title("GLOBAL (CELL 4)")
ax[1].imshow(cv2.cvtColor(local_u8, cv2.COLOR_BGR2RGB)); ax[1].axis("off")
ax[1].set_title("FUSED-BASED (CELL 5) — fused colours + detail maintained")
plt.tight_layout(); plt.show()

# the LAST / FINAL result on its own, full width
plt.figure(figsize=(18, 11))
plt.imshow(cv2.cvtColor(local_u8, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title("FINAL RESULT — fused-based HDR (colours & detail maintained)")
plt.show()

files.download("hdr_result_local.jpg")
print("done — hdr_result_local.jpg (the fused image, properly exposed — colours maintained)")
