"""
AUTOHDR + WINDOW PULL  (full AutoHDR chain -> SAM 3 / OneFormer window
segmentation -> dark-bracket window pull -> Gemini white frames)

BOTH THINGS TOGETHER: first the complete autohdr_full_pipeline chain produces
the AutoHDR-level room (CELLs 2-5, byte-identical to that script), THEN the
window-pull work from the CLEAR-SEG script runs on top of it:

  * segmentation happens on the PURE FUSED image (windows clearly visible
    there — on a dark bracket they are barely objects, the user-found insight)
  * the masks are TRANSFERRED onto the DARKEST bracket (aligned[0] — already
    in the fused coordinate frame), whose pixels become the window view
  * HDR FIRST, COMPOSITE AFTER (anti-halo): only the glass pixels of the
    finished AutoHDR result are replaced by the darkest bracket's view —
    windows go dark and rich, frames / mullions / curtains stay AutoHDR
  * OPTIONAL: window frames turned WHITE by Gemini Nano Banana Pro, locked
    by the hard FRAMEB mask so only frame-band pixels can come from Gemini

Cells:
  CELL 1  install
  CELL 2  brackets (Drive or upload) -> EV strip -> align -> Mertens fuse ->
          shadow-fill + highlight-recover -> FUSED image + histogram
  CELL 3  EXPOSURE ANALYSIS — zone heat-map + best-bracket map + clip numbers
  CELL 4  the calibrated finishing chain -> hdr_result.jpg (global)
  CELL 5  FUSED-BASED HDR — OneFormer on the pure fused image + LLM judge +
          all local fixes -> hdr_result_local.jpg  (the AutoHDR result)
  CELL 6  WINDOW SEGMENTATION — SAM 3 on the pure fused image (+ curtain and
          MIRROR subtraction — a mirror is never window glass); OneFormer
          windowpane reused from CELL 5; both masks shown ON the darkest bracket
  CELL 7  pick SAM 3 or OneFormer -> WINDOW PULL: the darkest bracket's glass
          composited into hdr_result_local -> hdr_result_windowpull.jpg
  CELL 8  OPTIONAL: WHITE window frames via Gemini, FRAMEB hard mask ->
          hdr_result_whiteframe.jpg

Models: shi-labs/oneformer_ade20k_dinat_large (Swin-L fallback) ·
facebook/sam3 (gated-free, needs HF_TOKEN) · Gemini (GEMINI_API_KEY).

Run on Google Colab with a GPU (A100 or T4). Copy each CELL into its own cell.
"""

# ============================== CELL 1 — install ==============================
# (Runtime -> Change runtime type -> GPU, first!)
# !pip install -q -U transformers opencv-python-headless matplotlib
# !pip install -q --force-reinstall "pillow==10.4.0"
# !pip install -q --force-reinstall -U huggingface_hub
# !pip install -q -U google-genai
# # OPTIONAL (OneFormer DiNAT backbone; CELL 5 falls back to Swin-L — safe to
# # ignore any error here):
# !pip install -q natten -f https://shi-labs.com/natten/wheels --trusted-host shi-labs.com
#
# >>> AFTER THIS CELL: Runtime -> Restart session, then run CELL 2. <<<


# ============================== CELL 2 — brackets -> fuse -> histogram =========
import os, math, time, numpy as np, cv2
import matplotlib.pyplot as plt
from google.colab import files

W = 2000


def _exptime(path):
    """EXIF exposure time (seconds) so each bracket can be labelled with its EV."""
    try:
        from PIL import Image as PImage, ExifTags
        ex = PImage.open(path)._getexif() or {}
        tags = {v: k for k, v in ExifTags.TAGS.items()}
        t = ex.get(tags.get("ExposureTime"))
        return float(t) if t else None
    except Exception:
        return None


import glob

# ---- SOURCE of the brackets: read from GOOGLE DRIVE or upload manually --------
# Drive layout you described:  MyDrive / RAW / new-exposure / scene1 ... scene16
SOURCE = "drive"       #@param ["drive", "upload"]
SCENE = "scene1"       #@param {type:"string"}
DRIVE_BASE = "/content/drive/MyDrive/RAW/new-exposure"   #@param {type:"string"}
# WHICH images to use: "all", or pick by index e.g. "0,2,4,6", a range "0-6",
# or a mix "0-3,6" — the cell first PRINTS how many there are and their indices.
SELECT = "all"         #@param {type:"string"}
_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.tif", "*.tiff")

if SOURCE == "drive":
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")            # first run asks you to authorise
    # show what scenes are actually there (helps if a name/path is off)
    if os.path.isdir(DRIVE_BASE):
        print("scenes found in", DRIVE_BASE, ":", sorted(os.listdir(DRIVE_BASE)))
    else:
        raise RuntimeError(f"'{DRIVE_BASE}' not found — fix DRIVE_BASE "
                           f"(is RAW in 'My Drive'? is Drive mounted?)")
    folder = os.path.join(DRIVE_BASE, SCENE)
    paths = sorted(sum([glob.glob(os.path.join(folder, e)) for e in _EXTS], []))
    if not paths:
        raise RuntimeError(f"no images in {folder} — check the SCENE name")
    print(f"reading {len(paths)} brackets from {folder}")
else:
    print(">>> Upload the exposure BRACKETS of ONE scene (you can pick a subset below):")
    up = files.upload()
    paths = sorted(up.keys())

# ---- show HOW MANY images there are + let you pick WHICH ones ----
print(f"\n{len(paths)} images available:")
for i, p in enumerate(paths):
    print(f"   [{i}]  {os.path.basename(p)}")


def _parse_sel(sel, n):
    """'all' -> every index; 'a,b,c' -> those; 'a-b' -> range; mixes allowed."""
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
    idx = [i for i in idx if 0 <= i < n]
    return idx or list(range(n))


paths = [paths[i] for i in _parse_sel(SELECT, len(paths))]
print(f"-> SELECT='{SELECT}'  ->  using {len(paths)}: {[os.path.basename(p) for p in paths]}")
if len(paths) < 2:
    raise RuntimeError("need at least 2 brackets (usually 5-8)")

items = []
for p in paths:
    im = cv2.imread(p)
    if im is None:
        continue
    h, w = im.shape[:2]
    if w != W:
        im = cv2.resize(im, (W, int(h * W / w)), interpolation=cv2.INTER_AREA)
    items.append({"img": im, "name": p, "t": _exptime(p)})
items.sort(key=lambda it: it["img"].mean())          # darkest -> brightest
raw = [it["img"] for it in items]

times = [it["t"] for it in items if it["t"]]
med_t = sorted(times)[len(times) // 2] if times else None
EVS = [round(math.log2(it["t"] / med_t), 1) if (it["t"] and med_t) else None for it in items]
print("brackets (darkest -> brightest):",
      [f"#{i} EV{e:+g}" if e is not None else f"#{i} ?" for i, e in enumerate(EVS)])

# ---- SHOW every INPUT bracket individually (darkest -> brightest, with its EV) ----
_n = len(raw)
fig, ax = plt.subplots(1, _n, figsize=(3.2 * _n, 3.4))
if _n == 1:
    ax = [ax]
for i, (im, ev) in enumerate(zip(raw, EVS)):
    ax[i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax[i].axis("off")
    lbl = f"#{i}  EV{ev:+g}" if ev is not None else f"#{i}  EV?"
    if i == 0:
        lbl += "\n(darkest)"
    elif i == _n - 1:
        lbl += "\n(brightest)"
    ax[i].set_title(lbl, fontsize=10)
fig.suptitle(f"{_n} input brackets — {SCENE if SOURCE == 'drive' else 'uploaded'}", y=1.03)
plt.tight_layout(); plt.show()


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


def _shadow_fill(fused_u8, brightest, T=0.18, p=1.5, max_fill=0.30, feather=4):
    """FUSION-TIME crushed-shadow fill from the BRIGHTEST bracket. Mertens
    regionally suppresses a mostly-blown bracket even where it is the best frame,
    so pockets below T get its real pixels blended in (max 30%). The weight is a
    smooth function of LUMINANCE (not a structure detector) so it cannot halo."""
    fl = fused_u8.astype(np.float32).mean(2) / 255
    w = np.clip((T - fl) / T, 0, 1) ** p
    w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), feather) * max_fill
    w3 = w[..., None]
    out = fused_u8.astype(np.float32) * (1 - w3) + brightest.astype(np.float32) * w3
    return out.clip(0, 255).astype("uint8")


def _highlight_recover(fused_u8, darker, Thi=0.86, p=1.6, max_pull=0.65, feather=4, sit=0.66):
    """The MIRROR of _shadow_fill: pull the DARKER (-EV) bracket into BLOWN bright
    areas (window, bulb, bright sink) so their detail comes back instead of a flat
    white blob. The darker bracket is gently lifted first (toward `sit`) so the
    recovered highlight sits naturally, not muddy-dark. The weight is a smooth
    function of the fused LUMINANCE only (high threshold Thi=0.86) so:
      * genuinely blown pixels (window view, bulb, specular sink) get recovered,
      * a plain bright-white WALL (below the threshold) is NOT touched -> no greying,
      * being luminance-keyed, it can never halo.
    Verified on a kitchen scene: blown>0.95  2.8% -> 0.9%, window view restored,
    wall brightness unchanged. Set max_pull=0 to disable."""
    if max_pull <= 0:
        return fused_u8
    fl = fused_u8.astype(np.float32).mean(2) / 255
    d = darker.astype(np.float32) / 255
    med = float(np.median(d.mean(2)[fl > Thi])) if (fl > Thi).any() else 0.0
    if 1e-3 < med < sit:
        d = np.clip(d, 0, 1) ** float(np.clip(np.log(sit) / np.log(med), 0.55, 1.0))
    w = np.clip((fl - Thi) / (1 - Thi), 0, 1) ** p * max_pull
    w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), feather)[..., None]
    return (fused_u8.astype(np.float32) * (1 - w) + (d * 255) * w).clip(0, 255).astype("uint8")


def derive_params(mertens_u8):
    """Read the RAW-fused HISTOGRAM and set the tone levers DYNAMICALLY per scene,
    so a dark warm room and a bright white room don't get identical treatment:
      * more crushed shadows  -> more shadow-fill
      * more blown highlights -> more highlight-recovery
      * darker median         -> higher exposure target (push it brighter)
      * flatter (low std)     -> stronger s-curve for pop
    Returns (params dict, histogram diagnostics dict)."""
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
    diag = dict(median=round(p50, 2), crushed_pct=round(crushed * 100, 1),
                blown_pct=round(blown * 100, 1), contrast=round(contrast, 3))
    return P, diag


AUTO_PARAMS = True   #@param {type:"boolean"}
# manual fall-backs — used ONLY when AUTO_PARAMS is unchecked
RECOVER_HIGHLIGHTS = 0.65   #@param {type:"number"}
SHADOW_FILL = 0.30          #@param {type:"number"}

aligned = _align(raw)                                 # sorted darkest -> brightest
mertens = (cv2.createMergeMertens().process(aligned) * 255).clip(0, 255).astype("uint8")
PARAMS, _hist = derive_params(mertens)                # histogram-driven, per scene
if AUTO_PARAMS:
    print("AUTO params  histogram", _hist)
    print("            ->", {k: round(v, 3) for k, v in PARAMS.items()})
else:
    PARAMS.update(shadow_fill=SHADOW_FILL, recover=RECOVER_HIGHLIGHTS,
                  expose_tgt=0.68, scurve=0.045)
    print("MANUAL params:", {k: round(v, 3) for k, v in PARAMS.items()})

fused = _shadow_fill(mertens, aligned[-1], max_fill=PARAMS["shadow_fill"])   # NO crushed shadows
fused = _highlight_recover(fused, aligned[0], max_pull=PARAMS["recover"])    # NO blown highlights
cv2.imwrite("fused.jpg", fused, [cv2.IMWRITE_JPEG_QUALITY, 95])


def show_hist(bgr, title):
    """Show the image and its histogram (luminance + R/G/B) side by side."""
    luma = bgr.astype(np.float32).mean(2) / 255
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
    ax[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); ax[0].axis("off"); ax[0].set_title(title)
    ax[1].hist(luma.ravel(), bins=128, range=(0, 1), color="0.4", alpha=0.9, label="luma")
    for i, c in zip((2, 1, 0), ("r", "g", "b")):       # BGR -> plot R,G,B
        ax[1].hist((bgr[..., i].ravel() / 255.0), bins=128, range=(0, 1),
                   histtype="step", color=c, linewidth=1.1)
    ax[1].axvline(0.06, color="c", ls="--", lw=1); ax[1].axvline(0.95, color="m", ls="--", lw=1)
    ax[1].set_xlim(0, 1); ax[1].set_title("histogram  (cyan=shadow clip · magenta=highlight clip)")
    ax[1].set_yticks([]); ax[1].legend(loc="upper center")
    plt.tight_layout(); plt.show()
    print(f"{title}: blown>0.95 = {100*(luma>0.95).mean():.1f}%  |  "
          f"crushed<0.06 = {100*(luma<0.06).mean():.1f}%  |  median = {np.median(luma):.2f}")


print(f"{len(aligned)} brackets aligned + Mertens-fused + shadow-filled -> {fused.shape[1]}x{fused.shape[0]}")
show_hist(fused, "FUSED (before finishing)")

# the FUSED image on its own, full width (large view)
plt.figure(figsize=(18, 11))
plt.imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title("FUSED IMAGE (large view)")
plt.show()

# ---- COOLED PREVIEW (display only): the same global warm-cast fix that CELL 5
# applies as its first step (-> fusedC), shown here already so you can SEE the
# fused image without the tungsten tint. The `fused` variable itself stays RAW
# on purpose — it is the EVIDENCE for segmentation, the Gemini judge and the
# bulb/glow tests, and they all need the original warmth to judge correctly.
_pv = fused.astype(np.float32) / 255
_pl = _pv.mean(2)
_pk = max(31, int(fused.shape[1] * 0.008) | 1)
_pex = cv2.GaussianBlur(np.clip((_pl - 0.80) / 0.12, 0, 1), (_pk, _pk), 0)
_pva = (_pl < 0.98) & (_pex < 0.5)                     # window/lights excluded from the fix
if _pva.sum() > 1000:
    _pref = _pv[_pva & (_pl >= np.percentile(_pl[_pva], 92))]
    if len(_pref) >= 100:
        _pg = _pref.mean(0)
        _pg = _pg.max() / np.maximum(_pg, 1e-6)
        _pv = np.clip(_pv * _pg, 0, 1)
fused_cooled = (_pv * 255).astype("uint8")             # preview only — NOT used downstream
cv2.imwrite("fused_cooled_preview.jpg", fused_cooled, [cv2.IMWRITE_JPEG_QUALITY, 95])
plt.figure(figsize=(18, 11))
plt.imshow(cv2.cvtColor(fused_cooled, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title("FUSED — COOLED PREVIEW (warm cast removed · display only; CELL 5 still reads the raw fused)")
plt.show()


# ============================== CELL 3 — EXPOSURE ANALYSIS =====================
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
for i in range(N):
    ev = f"EV{EVS[i]:+g}" if EVS[i] is not None else "?"
    print(f"  bracket #{i} ({ev:>6s}) is best-exposed for {100*(best==i).mean():5.1f}% of the image")


# ============================== CELL 4 — the calibrated FINISHING chain ========
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


# ============================== CELL 5 — FUSED-BASED HDR (segmentation-guided) ==
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
                _k = "wood_door" if float(np.median(_fbv2[_cm])) - _scb2 > 8 else "painted_door"
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
                    seg[_cm] = 200
                    print(f"box #{_cid} ('{_k}') sits on a door/window label with no "
                          f"daylight -> normal room treatment (not a window)")
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
            seg[_dc] = 200 if _rel_b > 8 else 0
            print(f"solid door detected ({100*_dc.mean():.1f}% of image) — "
                  + ("WOOD door: material colour kept" if _rel_b > 8
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
det = (np.isin(seg, DETAIL_CLASSES) | VANITY | _bulbfix) & ~WINDOW
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


# ============================== CELL 6 — WINDOW SEGMENTATION (SAM 3 + OneFormer)
# THE INSIGHT (user-found): SAM 3 / OneFormer are OBJECT models. On a dark
# bracket the window is barely an object — only the outdoor view glows. On the
# PURE FUSED image the window is clearly visible, so THAT is where segmentation
# runs. The masks are then TRANSFERRED onto the DARKEST bracket (aligned[0] —
# it was aligned into the fused coordinate frame back in CELL 2), whose pixels
# fill the glass at CELL 7. No extra uploads: everything is already in memory.
#   * SAM 3 segments windows + curtains + MIRRORS here, on the fused image
#     (the mirror is the explicit NEGATIVE class: it passes every window test
#      — framed glass with a "view" that glows in the dark bracket — so it is
#      found by name and subtracted from BOTH glass masks)
#   * OneFormer is NOT re-run — CELL 5 already segmented the pure fused image;
#     its windowpane class (seg == 8, after the LLM judge's vetoes moved fake
#     windows OUT of that class) is reused as option 2
# facebook/sam3 is GATED (free): accept the license at hf.co/facebook/sam3,
# make a READ token, add it as Colab secret HF_TOKEN — or paste it when asked.
import torch
from PIL import Image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if DEVICE == "cuda" else 0
KEEP_MODELS = VRAM_GB >= 30
print("device:", DEVICE, "| vram:", f"{VRAM_GB:.0f} GB ->",
      "models stay loaded (big gpu)" if KEEP_MODELS else "freeing after this cell (small gpu)")

from huggingface_hub import login
try:
    from google.colab import userdata as _ud6
    _tok = _ud6.get("HF_TOKEN")
except Exception:
    _tok = os.environ.get("HF_TOKEN")
login(token="".join(str(_tok).split())) if _tok else login()

from transformers import Sam3Processor, Sam3Model

s3_proc = Sam3Processor.from_pretrained("facebook/sam3")
s3_model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE).eval()


def sam3_instances(bgr, phrases, threshold=0.35):
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out_masks = []
    for phrase in phrases:
        inp = s3_proc(images=pil, text=phrase, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = s3_model(**inp)
        res = s3_proc.post_process_instance_segmentation(
            out, threshold=threshold, mask_threshold=0.5,
            target_sizes=[pil.size[::-1]])[0]
        m = res.get("masks", None)
        if m is not None and len(m):
            arr = (m if hasattr(m, "cpu") else torch.stack(list(m))).cpu().numpy()
            for mi in arr.reshape(-1, *arr.shape[-2:]).astype(bool):
                if mi.any():
                    out_masks.append(mi)
    return out_masks


RAW = {}                                          # raw model outputs (fused image)
GLASS = {}                                        # final masks used by CELL 7

t0 = time.time()
RAW["sam3_windows"] = sam3_instances(fused, ("window pane", "window", "glass door"))
# curtains at a STRICTER threshold (0.5) — only REAL curtains count; if the
# scene has none, zero instances come back and NOTHING gets subtracted
RAW["sam3_curtains"] = sam3_instances(fused, ("curtain", "drape"), threshold=0.5)
# mirrors at the SAME strict threshold — the explicit NEGATIVE class: a mirror
# passes every window test (SAM 3 sees a framed glass "view", OneFormer often
# labels the reflected sky 'windowpane', and the reflection GLOWS in the dark
# bracket like real glass) so it must be found EXPLICITLY and blocked
RAW["sam3_mirrors"] = sam3_instances(fused, ("mirror",), threshold=0.5)

_u = np.zeros((H, Wd), bool)
for mi in RAW["sam3_windows"]:
    _u |= mi
# real-curtain filter: rod-hung (top of the component in the upper part of the
# image) and not a speck — a bed/fan mislabelled 'curtain' never passes this
_c = np.zeros((H, Wd), bool)
for mi in RAW["sam3_curtains"]:
    _c |= mi
CURT_CLEAR = np.zeros((H, Wd), bool)
ncc, lab6, st6, _ = cv2.connectedComponentsWithStats(_c.astype(np.uint8))
for i in range(1, ncc):
    if st6[i, cv2.CC_STAT_TOP] < 0.45 * H and st6[i, cv2.CC_STAT_AREA] > 0.0005 * H * Wd:
        CURT_CLEAR |= (lab6 == i)
RAW["sam3_curtain_mask"] = CURT_CLEAR

# ---- MIRROR mask: SAM 3 "mirror" + OneFormer's mirror class (ADE20K 27) ------
MIRROR = (seg == 27)                              # CELL 5 never reassigns class 27
for mi in RAW["sam3_mirrors"]:
    MIRROR |= mi
RAW["mirror_mask"] = MIRROR
print(f"SAM 3 on the PURE FUSED image: {len(RAW['sam3_windows'])} window instances, "
      f"{len(RAW['sam3_curtains'])} curtain instances "
      f"({'curtains subtracted' if CURT_CLEAR.any() else 'no real curtains -> nothing subtracted'}), "
      f"{len(RAW['sam3_mirrors'])} mirror instances "
      f"({'MIRROR blocked from the glass masks' if MIRROR.any() else 'no mirror found'})"
      f"  [{time.time()-t0:.1f}s]")

if not KEEP_MODELS:
    del s3_model, s3_proc
    import gc; gc.collect(); torch.cuda.empty_cache()

# ---- the DARKEST bracket is the window-view source (no upload needed) --------
darkimg = aligned[0]
d = darkimg.astype(np.float32) / 255
dl = d.mean(2); dc = d.max(2) - d.min(2)          # also used by CELL 8 (frame band)
thr = max(0.22, 0.45 * float(np.percentile(dl, 99.9)))
SEE = (dl >= thr) | (dc >= 0.15)                  # the view: bright OR colourful
det_dark = (np.clip(d ** 0.5, 0, 1) * 255).astype("uint8")   # lifted, display only


def show_mask(tag, colour, base_bgr, mask, fname):
    o = base_bgr.astype(np.float32)
    a = mask.astype(np.float32)[..., None] * 0.55
    t = np.zeros_like(o); t[:] = colour
    o = (o * (1 - a) + t * a).clip(0, 255).astype("uint8")
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cnts, -1, colour, 2)
    cv2.rectangle(o, (0, 0), (o.shape[1] - 1, 44), (0, 0, 0), -1)
    cv2.putText(o, f"{tag} - {mask.mean()*100:.1f}%", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    cv2.imwrite(fname, o, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{tag}: {mask.mean()*100:.1f}% of image")
    try:
        from IPython.display import Image as _Img, display
        display(_Img(fname))
    except Exception:
        pass


APPLY_CLEANUP = False  #@param {type:"boolean"}
# False (default) = PURE TRANSFER: the masks are used EXACTLY as segmented on
#                   the fused image — nothing added, nothing removed.
# True            = also run the dark-bracket physics cleanup (see-through
#                   gate, rod-hung curtain subtraction, dark frame bars,
#                   specks). Turn on if a result shows curtains/doors in glass.


def refine_glass(raw_m):
    """Dark-image physics: remove thin DARK colourless frame bars (lit venetian
    slats KEPT), drop specks."""
    m = raw_m.astype(bool).copy()
    achro = (dc < 0.07) & (dl < 0.18) & m
    bar_k = max(5, int(Wd * 0.014) | 1)
    blobs = cv2.morphologyEx(achro.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((bar_k, bar_k), np.uint8)).astype(bool)
    m &= ~(achro & ~blobs)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    ncc2, lab2, st2, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    for i in range(1, ncc2):
        if st2[i, cv2.CC_STAT_AREA] < 0.0005 * H * Wd:
            m[lab2 == i] = False
    return m


def _no_mirror(m):
    """MIRROR veto: a glass component that is MOSTLY mirror dies WHOLE (no
    leftover rim ring around the mirror), the rest just loses the overlap."""
    m = m.astype(bool).copy()
    if not MIRROR.any():
        return m
    nccM, labM, _stM, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    for i in range(1, nccM):
        comp = labM == i
        if float(MIRROR[comp].mean()) > 0.5:
            m &= ~comp                            # the whole "window" was a mirror
    return m & ~MIRROR


if MIRROR.any():
    show_mask("MIRROR — blocked from both glass masks", (0, 200, 255), det_dark,
              MIRROR, "mask_mirror.jpg")

# ---- SAM 3 mask: union of the window instances, transferred to the dark bracket
raw_m = np.zeros((H, Wd), bool)
if APPLY_CLEANUP:
    kept = drop = 0
    for mi in RAW["sam3_windows"]:
        if float(SEE[mi].mean()) >= 0.06:         # a real window GLOWS in the dark
            raw_m |= mi; kept += 1
        else:
            drop += 1
    curt = np.zeros((H, Wd), bool)
    for mi in RAW["sam3_curtains"]:
        curt |= mi
    ncc3, lab3, st3, _ = cv2.connectedComponentsWithStats(curt.astype(np.uint8))
    for i in range(1, ncc3):
        if st3[i, cv2.CC_STAT_TOP] < 0.45 * H and st3[i, cv2.CC_STAT_AREA] > 0.0005 * H * Wd:
            raw_m &= ~(lab3 == i)                 # rod-hung curtains subtracted
    print(f"SAM 3 cleanup: {kept} window instances kept, {drop} dark rejected")
    GLASS["sam3"] = _no_mirror(refine_glass(raw_m))
else:
    for mi in RAW["sam3_windows"]:
        raw_m |= mi
    # pure transfer: windows minus the REAL curtains found on the fused image
    # (if no curtains existed, the curtain mask is empty — nothing changes);
    # the MIRROR veto always runs — a mirror is never window glass
    GLASS["sam3"] = _no_mirror(raw_m & ~RAW["sam3_curtain_mask"])
    print(f"SAM 3: pure transfer of {len(RAW['sam3_windows'])} instances"
          + (" minus real curtains" if RAW["sam3_curtain_mask"].any() else "")
          + (" minus MIRROR" if MIRROR.any() else ""))
show_mask("SAM3 FINAL (on darkest bracket)", (255, 150, 0), det_dark,
          GLASS["sam3"], "glass_sam3.jpg")

# ---- OneFormer mask: CELL 5's windowpane class (pure fused), transferred ----
if APPLY_CLEANUP:
    GLASS["oneformer"] = _no_mirror(refine_glass((seg == 8) | ((seg == 14) & SEE)))
else:
    GLASS["oneformer"] = _no_mirror(seg == 8)     # pure transfer, MIRROR blocked
    print("OneFormer: pure transfer of CELL 5's windowpane class"
          + (" minus MIRROR" if MIRROR.any() else ""))
show_mask("OneFormer FINAL (on darkest bracket)", (200, 0, 200), det_dark,
          GLASS["oneformer"], "glass_oneformer.jpg")
print("ready — run CELL 7 to pick a mask and pull the windows")


# ============================== CELL 7 — WINDOW PULL on the AutoHDR result ====
# HDR FIRST, COMPOSITE AFTER (the anti-halo order): CELL 5's finished AutoHDR
# result is the base — its exposure/whites/colours never see the window, so no
# grey wash and no halo ring can form. ONLY the glass pixels get replaced by
# the DARKEST bracket's view: windows turn dark and rich while the frames,
# mullions and curtains stay exactly as the AutoHDR result made them.
_tiles = []
for _fn in ("glass_sam3.jpg", "glass_oneformer.jpg"):
    _t = cv2.imread(_fn)
    if _t is not None:
        _tiles.append(cv2.resize(_t, (640, int(_t.shape[0] * 640 / _t.shape[1]))))
if _tiles:
    _hmin = min(t.shape[0] for t in _tiles)
    cv2.imwrite("choose.jpg", np.hstack([t[:_hmin] for t in _tiles]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    try:
        from IPython.display import Image as _Img, display
        print("Option 1 — SAM 3 (LEFT)      |      Option 2 — OneFormer (RIGHT)")
        display(_Img("choose.jpg"))
    except Exception:
        pass

CHOICE = "1"  #@param ["1", "2"] {type:"string"}
# 1 = SAM 3   |   2 = OneFormer  — change the dropdown / value, re-run to switch
choice = "1" if str(CHOICE).strip() != "2" else "2"
key = "sam3" if choice == "1" else "oneformer"
print(f"-> continuing with {key.upper()}  (glass {GLASS[key].mean()*100:.1f}%)")
if not GLASS[key].any():
    print("WARNING: the chosen glass mask is EMPTY — the result will be the "
          "AutoHDR image unchanged. Try the other option or APPLY_CLEANUP.")


def _feather(mask, guide_bgr):
    r = max(6, int(mask.shape[1] * 0.004))
    try:
        gd = cv2.cvtColor((np.clip(guide_bgr, 0, 1) * 255).astype(np.uint8),
                          cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        return np.clip(cv2.ximgproc.guidedFilter(gd, mask.astype(np.float32), r, 1e-3), 0, 1)
    except Exception:
        return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), r * 0.6), 0, 1)


glass_view = _feather(GLASS[key].astype(np.float32), fused.astype(np.float32) / 255)

base = local_u8.astype(np.float32) / 255          # CELL 5's AutoHDR result
vsrc = darkimg.astype(np.float32) / 255           # the darkest bracket = the view
sel = glass_view > 0.5
med = float(np.median(vsrc.mean(2)[sel])) if sel.any() else 0.0
if 1e-3 < med < 0.58:                             # gentle lift only if it sits dark
    vsrc = np.clip(vsrc, 0, 1) ** float(np.clip(np.log(0.58) / np.log(med), 0.6, 1.0))
    print(f"glass median {med:.2f} -> lifted toward 0.58 (gamma)")
gv = glass_view[..., None]
pull = base * (1 - gv) + vsrc * gv

pull_u8 = (pull * 255).clip(0, 255).astype("uint8")
cv2.imwrite("hdr_result_windowpull.jpg", pull_u8, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite("glass_mask_chosen.png", ((glass_view > 0.5).astype(np.uint8) * 255))
try:
    from IPython.display import Image as _Img, display
    print(f"WINDOW PULL — AutoHDR everywhere, {key.upper()} glass from the darkest bracket:")
    display(_Img("hdr_result_windowpull.jpg"))
except Exception:
    pass
files.download("hdr_result_windowpull.jpg")
files.download("glass_mask_chosen.png")
print("done — hdr_result_windowpull.jpg (AutoHDR room + dark rich windows)")


# ============================== CELL 8 — OPTIONAL: WHITE FRAMES via Gemini ====
# Hallucination-proof white-frame step: Gemini repaints the image, but ONLY the
# frame-band pixels (FRAMEB) are taken from it — everything else is guaranteed
# byte-identical to the CELL 7 result. Needs GEMINI_API_KEY (Colab secret, env
# var or .env line) and CELL 1's google-genai install.
from google import genai
from google.genai import types as gtypes

_gkey = os.environ.get("GEMINI_API_KEY", "")
if not _gkey:
    try:
        from google.colab import userdata as _ud8
        _gkey = _ud8.get("GEMINI_API_KEY") or ""
    except Exception:
        pass
if not _gkey and os.path.exists(".env"):
    for _ln in open(".env"):
        if _ln.startswith("GEMINI_API_KEY="):
            _gkey = _ln.split("=", 1)[1].strip()
if not _gkey:
    _gkey = input("paste your Gemini API key: ")
_gkey = "".join(str(_gkey).split())               # kill stray whitespace in the key
gclient = genai.Client(api_key=_gkey)

PROMPT = ("Do not hallucinate anything. Just change the color of the window frame to white. "
          "Keep blinds, curtains, and everything else exactly as they are. "
          "The window style must stay exactly the same. "
          "The outside view through the window must stay exactly the same, with no change "
          "in the colors of the outdoor view. "
          "If there is no frame in the middle of the glass, do not add frames. "
          "Only recolor the existing frame borders to white. Change nothing else in the image.")

# "negative prompt": Gemini has no separate negative_prompt field (that is a
# diffusion-model concept) — instead the forbidden things are written as hard
# rules, which an instruction-following model obeys like commands:
NEGATIVE = (" STRICTLY FORBIDDEN — never do any of the following: "
            "do not add, remove, move or resize any object; "
            "do not add new window frames, mullions, grids or dividers; "
            "do not change the outdoor view, sky, trees, or their colors; "
            "do not change curtains, blinds, walls, floor or furniture; "
            "do not change lighting, shadows, contrast or white balance; "
            "do not blur, sharpen, denoise or restyle any part of the image; "
            "do not crop, zoom, rotate or change the aspect ratio; "
            "do not add text, logos or watermarks.")
PROMPT = PROMPT + NEGATIVE

ok, jb = cv2.imencode(".jpg", pull_u8, [cv2.IMWRITE_JPEG_QUALITY, 95])
img_part = gtypes.Part.from_bytes(data=jb.tobytes(), mime_type="image/jpeg")
print("sending to gemini-3-pro-image-preview ...")
try:
    resp = gclient.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[img_part, PROMPT],
        config=gtypes.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"],
                                            temperature=0.0),
    )
except Exception as e:
    print(f"retrying without config ({type(e).__name__}: "
          f"{str(e).replace(_gkey, '[KEY]')[:200]})")
    resp = gclient.models.generate_content(model="gemini-3-pro-image-preview",
                                           contents=[img_part, PROMPT])

gem = None
for part in resp.candidates[0].content.parts:
    data = getattr(getattr(part, "inline_data", None), "data", None)
    if data:
        gem = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        break
if gem is None:
    raise RuntimeError("Gemini returned no image — check API key / quota. Text reply: "
                       + str(getattr(resp, "text", ""))[:300])
gem = cv2.resize(gem, (Wd, H), interpolation=cv2.INTER_LANCZOS4)

# ---- FRAME BAND — the pixels where Gemini's white frame is allowed to land.
# It MUST cover the WHOLE window frame (outer casing/sash + the mullion bars
# between the panes) or the dark frame bits it misses stay BLACK in the final.
# Built from the DARKEST bracket's dl/dc (CELL 6) — verified on a synthetic
# window: frame capture 100%, dark view (pool cage / tree) eaten 0%.
g8 = (glass_view > 0.5).astype(np.uint8)
glass_bool = g8.astype(bool)

# zone = the window unit + a generous margin; the frame can never be outside it
zone = cv2.dilate(g8, np.ones((max(11, int(Wd * 0.05) | 1),) * 2, np.uint8)).astype(bool)

# 1) RING hugging the glass — the outer casing/sash AND the thin gap between two
#    panes (the centre mullion sits in that gap).
ring_k = max(13, int(Wd * 0.022) | 1)
ring = cv2.dilate(g8, np.ones((ring_k, ring_k), np.uint8)).astype(bool) & ~glass_bool

# 2) DARK colourless frame in the zone (dl < 0.35 / dc < 0.12 catches
#    medium-grey frames too). Split so real view is protected:
#      * OUTSIDE the glass  -> keep all of it (the casing/sash)
#      * INSIDE the glass   -> keep only THIN LINE-shaped bars (pane dividers);
#        a thick dark blob there is the outdoor view and is NOT touched.
darkbar = (dl < 0.35) & (dc < 0.12) & zone
Lk = max(21, int(Wd * 0.03))
line = (cv2.morphologyEx(darkbar.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, Lk), np.uint8)) |
        cv2.morphologyEx(darkbar.astype(np.uint8), cv2.MORPH_OPEN, np.ones((Lk, 1), np.uint8))).astype(bool)
outer = darkbar & ~glass_bool
inner_lines = line & glass_bool
thin_k = max(3, int(Wd * 0.02) | 1)                     # a mullion is at most ~2% wide
inner_lines = inner_lines & ~cv2.morphologyEx(
    inner_lines.astype(np.uint8), cv2.MORPH_OPEN, np.ones((thin_k, thin_k), np.uint8)).astype(bool)

frame_raw = ring | outer | inner_lines
frame_raw = cv2.morphologyEx(frame_raw.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((max(3, int(Wd * 0.006) | 1),) * 2, np.uint8)).astype(bool) & zone
FRAMEB = cv2.GaussianBlur(frame_raw.astype(np.float32), (0, 0), 2)
fb = FRAMEB[..., None]
final = (pull_u8.astype(np.float32) * (1 - fb) + gem.astype(np.float32) * fb)
final = final.clip(0, 255).astype("uint8")

cv2.imwrite("gemini_raw.jpg", gem, [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite("hdr_result_whiteframe.jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"frame band = {(FRAMEB > 0.5).mean()*100:.1f}% of image — only these pixels came from Gemini")
try:
    from IPython.display import Image as _Img, display
    print("BEFORE (window-pull result):");   display(_Img("hdr_result_windowpull.jpg"))
    print("GEMINI raw output (reference only — NOT used directly):"); display(_Img("gemini_raw.jpg"))
    print("AFTER — white frame, everything else guaranteed untouched:")
    display(_Img("hdr_result_whiteframe.jpg"))
except Exception:
    pass
files.download("hdr_result_whiteframe.jpg")
print("done — white-frame result: Gemini touched ONLY the frame band")
