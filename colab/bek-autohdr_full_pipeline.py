"""
AUTOHDR-LEVEL FULL PIPELINE  (brackets -> analysis -> calibrated HDR -> optional
segmentation-guided local exposure)

Goal: a result at the level of AutoHDR / autoenhance.ai —
  * NO crushed shadows      (fusion-time fill from the brightest bracket)
  * NO halos / "hollows"     (every mask is luminance-keyed or edge-blurred)
  * natural light            (mixed-light casts neutralised)
  * NO wash-out              (exposure metered on the room, window shielded)
  * smooth wall colours      (neutralise + halo-band + bulb-glow fixes)
  * highlights MAINTAINED     (soft-shoulder levels — near-white keeps detail)
  * just a PINCH of warmth     (small mid-tone warmth so it is natural, not clinical)

Cells:
  CELL 1  install
  CELL 2  upload BRACKETS -> EV strip -> align -> Mertens fuse -> shadow-fill
          -> show the FUSED image + its HISTOGRAM (luma + R/G/B)
  CELL 3  EXPOSURE ANALYSIS — a zone heat-map (what needs +/- exposure) and a
          best-bracket map (what each area 'wants') + clipping numbers
  CELL 4  the calibrated finishing chain -> AUTOHDR-LEVEL result, with the
          BEFORE/AFTER histogram side by side. Download.
  CELL 5  OPTIONAL "HDR implementation" — segmentation-guided LOCAL exposure:
          OneFormer segments wall/ceiling/floor/window, each region is measured
          against a target and gently pushed to it (per-region ±EV table shown),
          so corrections land only where needed instead of on the whole image.

Run on Google Colab with a GPU (A100 / T4). Copy each CELL into its own cell.
"""

# ============================== CELL 1 — install ==============================
# (Runtime -> Change runtime type -> GPU, first!)
# !pip install -q -U transformers opencv-python-headless matplotlib
# !pip install -q --force-reinstall "pillow==10.4.0"
# !pip install -q --force-reinstall -U huggingface_hub
# # OPTIONAL (only for CELL 5's OneFormer DiNAT backbone; it auto-falls back to
# # Swin-L if this fails — safe to ignore any error here):
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


# ============================== CELL 5 — OPTIONAL: segmentation-guided LOCAL exposure
# The "HDR implementation option": instead of one global exposure, OneFormer
# segments the room into wall/ceiling, floor and window; each region is measured
# against a natural target and gently pushed to it (feathered per-region gamma).
# A ±EV table is printed so you can see exactly what got adjusted and by how much.
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
# SEGMENT ON THE FUSED IMAGE (not the whitened result): the fused image still
# shows every material's TRUE colour — golden vanity wood, marble tile veins,
# stainless steel — so OneFormer separates the classes far more reliably there.
# The enhancements are then applied to the result, guided by this segmentation
# and by the fused image's real material colours.
pil = Image.fromarray(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
inp = of_proc(images=pil, task_inputs=["semantic"], return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = of_model(**inp)
seg = of_proc.post_process_semantic_segmentation(out, target_sizes=[pil.size[::-1]])[0].cpu().numpy()

# ---- SHOW the segmentation as a colour mask so you can SEE what was detected ----
# each region gets its own colour tinted over the result; % of image printed.
SEG_COLOURS = {                                        # name: (classes, BGR colour)
    "wall/ceiling": ((0, 5, 1),         (255, 120, 0)),   # blue
    "floor/rug":    ((3, 28, 13),       (0, 200, 0)),     # green
    "table/furn":   ((15, 33, 44, 35),  (0, 140, 255)),   # orange
    "sink/basin":   ((47, 37, 65),      (140, 90, 255)),  # pink — sink, bathtub, toilet
    "appliance":    ((71, 50, 118, 124, 129, 107, 133), (200, 0, 200)),  # magenta — stove/fridge/oven/…
    "window/door":  ((8, 14),           (255, 255, 0)),   # cyan
}
# NOTE: kitchen CABINETS (10), SHELVES (24) and COUNTERS (45) are deliberately NOT
# handled here — they are neither shown nor adjusted, so they stay exactly as the
# CELL-4 chain made them. To bring them back, add a row like:
#   "cabinet/shelf": ((10, 24, 45), (200, 0, 200)),      # magenta
# and add those classes to REGIONS below.
# ---- BULBS + their GLOW halo (the warm reflection spreading around a light) ----
# OneFormer's lamp classes catch fixtures; a bright + warm blob catches bulbs it
# missed (recessed cans, pendant bulbs). GLOW = a dilated halo around them that
# actually landed on the white ceiling/wall — that is the warm reflection we clean.
_lb = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
_Lr = _lb[..., 0] / 255; _ar = _lb[..., 1] - 128; _brr = _lb[..., 2] - 128
_chr = np.sqrt(_ar * _ar + _brr * _brr)
BULB = np.isin(seg, (36, 82, 85)) | ((_Lr > 0.86) & (_brr > 5))   # lamp/sconce/chandelier OR bright-warm blob
BULB = cv2.morphologyEx(BULB.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
_gk = max(9, int(Wd * 0.035) | 1)
GLOW = cv2.dilate(BULB.astype(np.uint8), np.ones((_gk, _gk), np.uint8)).astype(bool)
GLOW = GLOW & ~BULB & (_Lr > 0.60) & (_brr > 3) & (_chr < 18) & np.isin(seg, (0, 5, 1))

_ov = result.astype(np.float32).copy()
print("segments detected:")
for _nm, (_cls, _col) in SEG_COLOURS.items():
    _mk = np.isin(seg, _cls)
    if _mk.mean() < 0.001:
        print(f"  {_nm:14s}  (not found)"); continue
    _a = _mk.astype(np.float32)[..., None] * 0.45
    _t = np.zeros_like(_ov); _t[:] = _col
    _ov = _ov * (1 - _a) + _t * _a
    print(f"  {_nm:14s}  {100*_mk.mean():4.1f}% of image")
for _nm, _msk, _col, _al in (("glow halo", GLOW, (0, 220, 255), 0.4),
                             ("bulbs", BULB, (0, 255, 255), 0.6)):   # yellow
    if _msk.mean() < 0.0005:
        print(f"  {_nm:14s}  (not found)"); continue
    _a = _msk.astype(np.float32)[..., None] * _al
    _t = np.zeros_like(_ov); _t[:] = _col
    _ov = _ov * (1 - _a) + _t * _a
    print(f"  {_nm:14s}  {100*_msk.mean():4.2f}% of image")
_ov = _ov.clip(0, 255).astype("uint8")
cv2.imwrite("segmentation_mask.jpg", _ov, [cv2.IMWRITE_JPEG_QUALITY, 92])
plt.figure(figsize=(18, 11))
plt.imshow(cv2.cvtColor(_ov, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title("SEGMENTATION — blue wall/ceiling · green floor · orange table · magenta cabinet · cyan window · yellow bulb+glow")
plt.show()

# ADE20K classes -> regions, each with a natural luminance target.
# WALL and CEILING are SPLIT on purpose — measured on the AutoHDR reference the
# ceiling sits at ~0.83 but the walls at only ~0.77; one combined 0.82 target
# pushed the walls to ~0.88 (too bright) and drove the blown-highlight count up.
REGIONS = {
    "ceiling":      {"cls": (5,),             "target": 0.83},   # ceiling
    "wall":         {"cls": (0, 1),           "target": 0.77},   # wall, building
    "floor":        {"cls": (3, 28, 13),      "target": 0.52},   # floor, rug, ground
    "table/desk":   {"cls": (15, 33),         "target": 0.60},   # table, desk (NOT cabinet/shelf)
    # A white sink / basin / tub is a WHITE OBJECT, not a wall: pushed to the wall
    # target it blows out and loses its shape. A slightly LOWER target keeps its
    # form and detail (this is the "sink area too bright" fix).
    "sink/basin":   {"cls": (47, 37, 65),     "target": 0.76},   # sink, bathtub, toilet
}
WINDOW = np.isin(seg, (8, 14))                          # window/glass door -> never touched
resf = result.astype(np.float32) / 255
lu = resf.mean(2)

print("region              current   target   suggested")
adjust = np.ones((H, Wd), np.float32)                   # per-pixel gamma map (1 = no change)
for name, spec in REGIONS.items():
    m = np.isin(seg, spec["cls"]) & ~WINDOW
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
local = _pinch_warm(local, 0)                           # (no extra warmth; already applied)
local_u8 = (local * 255).clip(0, 255).astype("uint8")

# ---- RESTORE the natural COLOUR of the TABLE / free-standing furniture ONLY -----
# A low-chroma warm wood surface (a dining table, a vanity/dresser) looks like an
# off-white WALL to the global chain, so it gets neutralised toward bright white and
# loses its tone. Here we bring the real colour (LAB a,b) back from the FUSED image
# inside the segmented TABLE, keeping the finished BRIGHTNESS (L) — so the piece reads
# natural wood instead of bleached white, without going dark.
# NOTE: this is TABLE / free-standing furniture ONLY — kitchen CABINETS, shelves and
# counters are deliberately EXCLUDED (they stay as the chain made them; in a white
# kitchen they stay clean white). Add 10 (cabinet) / 45 (counter) to TABLE_CLASSES
# only if you also want wooden cabinets restored.
RESTORE_COLOR = 0.7   #@param {type:"number"}   # 0 = off, 0.7 = default, 1 = full
TABLE_CLASSES = (15, 33, 44, 35)   # table, desk, chest-of-drawers, wardrobe
                                   # (NOT 10 cabinet · NOT 24 shelf · NOT 45 counter)


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
if RESTORE_COLOR > 0 and wood.sum() > 0.002 * H * Wd:
    r_lab = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    f_lab = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = cv2.GaussianBlur(wood.astype(np.float32), (0, 0), 4)[..., None] * RESTORE_COLOR
    r_lab[..., 1:] = r_lab[..., 1:] * (1 - m) + f_lab[..., 1:] * m   # a,b from fused; L kept
    local_u8 = cv2.cvtColor(np.clip(r_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"restored natural colour on {100*wood.mean():.1f}% of the image (TABLE / furniture only)")
else:
    print("no table/furniture region found to restore (or RESTORE_COLOR = 0)")

# ---- MATERIAL RESTORE (appearance-conditioned): real wood keeps its colour -----
# The FUSED image is the ground truth for what each surface REALLY is: a wooden
# bathroom vanity is strongly warm there (measured b+24, same as AutoHDR keeps it)
# while a white kitchen cabinet is near-neutral (b+3..6). So for cabinet/furniture
# classes, restore the fused colour ONLY where the fused image says "this is real
# warm material" — white cabinets stay clean white, wooden vanities get their
# golden wood back. (This fixes the bleached bathroom vanity: b+10.6 -> ~+21.)
MATERIAL_RESTORE = 0.85   #@param {type:"number"}   # 0 = off, 0.85 = default
MATERIAL_CLASSES = (10, 24, 44, 35, 45)   # cabinet, shelf, chest, wardrobe, counter
matcls = np.isin(seg, MATERIAL_CLASSES) & ~WINDOW
if MATERIAL_RESTORE > 0 and matcls.sum() > 0.002 * H * Wd:
    f_lab = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB).astype(np.float32)
    f_b = f_lab[..., 2] - 128
    # the appearance gate is RELATIVE to the scene's own CEILING: the fused image
    # is pre-white-balance, so in a warm-lit kitchen even WHITE cabinets read b+20
    # — but so does the ceiling (same cast). Real wood is far warmer than the
    # scene's own painted whites (measured: bathroom vanity +24 over its ceiling,
    # white kitchen cabinet only +7 over its). The ceiling is the safest neutral
    # reference — always painted, never tiled/wooden; walls are the fallback.
    _ceil = seg == 5
    if _ceil.sum() > 1000:
        _ref = float(np.median(f_b[_ceil]))
    else:
        _wallpix = np.isin(seg, (0, 1))
        _ref = float(np.percentile(f_b[_wallpix], 35)) if _wallpix.sum() > 1000 else 0.0
    warm_gate = np.clip((f_b - _ref - 12) / 8, 0, 1)
    m = cv2.GaussianBlur((matcls.astype(np.float32) * warm_gate), (0, 0), 4)[..., None] * MATERIAL_RESTORE
    r_lab = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_lab[..., 1:] = r_lab[..., 1:] * (1 - m) + f_lab[..., 1:] * m   # colour from fused; L kept
    local_u8 = cv2.cvtColor(np.clip(r_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    _hit = float((matcls & (warm_gate > 0.5)).mean())
    print(f"material restore: real wood colour returned on {100*_hit:.1f}% "
          f"(white cabinets untouched — appearance-gated on the fused image)")

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
        f_lab = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_a = float(np.clip(np.median((f_lab[..., 1] - 128)[floor]), 0.0, 4.0))
        tgt_b = float(np.clip(np.median((f_lab[..., 2] - 128)[floor]), 4.0, 10.0))
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

# ---- CLEAN the wall/ceiling COLOUR: force the segmented walls + ceiling neutral,
# so any residual warm cast the global chain missed (e.g. the tan SOFFIT band above
# the cabinets) and any stray pinch-warmth on a white surface are removed -> smooth
# clean white. The mask is DILATED a little so it also catches the thin soffit strip
# OneFormer often mislabels; wood/table, floor and window are all excluded.
NEUTRALIZE_WALLS = 0.9   #@param {type:"number"}   # 0 = off, 0.9 = default, 1 = fully neutral
wc = np.isin(seg, (0, 5, 1))
wc = cv2.dilate(wc.astype(np.uint8), np.ones((max(3, int(Wd * 0.008) | 1),) * 2, np.uint8)).astype(bool)
wc = wc & ~WINDOW & ~wood & ~floor                 # never grey the window / table / floor
if NEUTRALIZE_WALLS > 0 and wc.sum() > 0.002 * H * Wd:
    # VEIN PROTECTION: marble-tile veining is warm STRUCTURE (thin, high-frequency
    # — its fused warmth stands out from its neighbourhood), while a tungsten cast
    # is a warm smooth FIELD (its high-pass is ~0). Sparing only the warm
    # high-frequency part keeps the tile veins marble while the soffit cast still
    # gets cleaned to white.
    f_lab = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB).astype(np.float32)
    f_b = f_lab[..., 2] - 128
    vein = np.clip((f_b - cv2.GaussianBlur(f_b, (0, 0), 25) - 3) / 5, 0, 1)
    lab = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = cv2.GaussianBlur(wc.astype(np.float32), (0, 0), 4) * NEUTRALIZE_WALLS * (1 - vein)
    lab[..., 1] = lab[..., 1] * (1 - m) + 128 * m   # a -> neutral
    lab[..., 2] = lab[..., 2] * (1 - m) + 128 * m   # b -> neutral
    local_u8 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"neutralised wall/ceiling colour on {100*wc.mean():.1f}% "
          f"(smooth white; tile veining spared on {100*float((wc & (vein > 0.5)).mean()):.1f}%)")

# ---- CLEAN the BULB GLOW: neutralise the warm reflection halo around each light
# so the ceiling/wall around a bulb stays clean white (the bulb core itself is left
# alone — it keeps its natural glow). The GLOW mask was found above; here we just
# pull its a,b toward neutral, feathered so there is no ring/edge.
NEUTRALIZE_GLOW = 0.85   #@param {type:"number"}   # 0 = off, 0.85 = default
if NEUTRALIZE_GLOW > 0 and GLOW.sum() > 0.0005 * H * Wd:
    lab = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = cv2.GaussianBlur(GLOW.astype(np.float32), (0, 0), 6) * NEUTRALIZE_GLOW
    lab[..., 1] = lab[..., 1] * (1 - m) + 128 * m
    lab[..., 2] = lab[..., 2] * (1 - m) + 128 * m
    local_u8 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"neutralised bulb-glow halo on {100*GLOW.mean():.2f}% (clean ceiling around lights)")

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
    metal = (metal | _box) & ~WINDOW & ~floor & ~wood   # never touch floor/table
if NEUTRAL_METALS > 0 and metal.sum() > 0.001 * H * Wd:
    lab = cv2.cvtColor(local_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    ma = lab[..., 1] - 128; mb = lab[..., 2] - 128
    mch = np.sqrt(ma * ma + mb * mb)
    gate = np.clip((28 - mch) / 10, 0, 1)            # full neutralize up to ch~18; a genuinely
                                                     # coloured object (red kettle, ch 40+) stays
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
    _knee = 0.84
    _slope = 1 - 0.45 * SUN_TONE
    _new = np.where(_lu > _knee, _knee + (_lu - _knee) * _slope, _lu)
    _ratio = np.where(_lu > 1e-6, _new / np.maximum(_lu, 1e-6), 1.0)
    _keep = cv2.GaussianBlur(((seg == 5) | BULB).astype(np.float32), (0, 0), 5)  # ceiling+bulbs stay
    _ratio = 1.0 + (_ratio - 1.0) * (1.0 - _keep)
    local_u8 = (np.clip(_f * _ratio[..., None], 0, 1) * 255).astype("uint8")
    print(f"sun-tone {SUN_TONE}: sunlit near-white areas toned toward AutoHDR level (ceiling/bulbs kept)")

# final highlight safety: the per-region gammas can push a few pixels back over
# the knee — re-apply the soft compressor so nothing clips to paper white.
local_u8 = (_soft_highlights(local_u8.astype(np.float32) / 255) * 255).clip(0, 255).astype("uint8")

cv2.imwrite("hdr_result_local.jpg", local_u8, [cv2.IMWRITE_JPEG_QUALITY, 95])

# histograms (numbers)
show_hist(result, "GLOBAL result (CELL 4)")
show_hist(local_u8, "LOCAL region-tuned result (CELL 5)")

# GLOBAL vs LOCAL side by side (big) so you can compare the two finals
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
ax[0].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); ax[0].axis("off")
ax[0].set_title("GLOBAL (CELL 4)")
ax[1].imshow(cv2.cvtColor(local_u8, cv2.COLOR_BGR2RGB)); ax[1].axis("off")
ax[1].set_title("LOCAL region-tuned (CELL 5)")
plt.tight_layout(); plt.show()

# the LAST / FINAL result on its own, full width
plt.figure(figsize=(18, 11))
plt.imshow(cv2.cvtColor(local_u8, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title("FINAL RESULT  (segmentation-guided local exposure)")
plt.show()

files.download("hdr_result_local.jpg")
print("done — hdr_result_local.jpg (each region nudged to its natural target)")


# ============================== CELL 6 — AUTO-CALIBRATE (reference-free) ========
# Normalise EVERY result to the same AutoHDR-style targets — NO reference image
# needed. It gently, GLOBALLY corrects three things so every scene lands
# consistent and clean:
#   1) BRIGHTNESS  — median luminance -> TARGET_MEDIAN (gentle gamma, clamped)
#   2) WHITE BALANCE — the bright pixels are pulled to NEUTRAL (clean whites)
#   3) SATURATION  — only EXTREMES are corrected: a too-flat scene gets a mild
#      boost, an over-saturated one a mild tame; a naturally-coloured scene is
#      left alone (so a colourful bathroom keeps its colour, a white kitchen
#      stays clean). All three are GLOBAL monotonic transforms, so they can NOT
#      create halos, black dots or local shades.
TARGET_MEDIAN = 0.73   #@param {type:"number"}   # room brightness (0.70 natural .. 0.76 bright)
WB_STRENGTH = 0.7      #@param {type:"number"}   # how hard to neutralise the whites
AUTO_CALIBRATE = True  #@param {type:"boolean"}

_base = local_u8 if "local_u8" in globals() else result   # prefer the CELL-5 output


def _stats(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] / 255; a = lab[..., 1] - 128; b = lab[..., 2] - 128
    br = L > 0.75; ch = np.sqrt(a * a + b * b); col = ch > 4
    wa = float(a[br].mean()) if br.sum() > 500 else 0.0
    wb = float(b[br].mean()) if br.sum() > 500 else 0.0
    cc = float(np.median(ch[col])) if col.sum() > 500 else 0.0
    return dict(median=round(float(np.median(L)), 2), white_a=round(wa, 1),
                white_b=round(wb, 1), chroma=round(cc, 1))


def auto_finish(img_u8, target_median=0.73, wb_strength=0.7):
    """Reference-free normalisation to consistent AutoHDR-style targets."""
    img = img_u8.astype(np.float32) / 255
    med = float(np.median(img.mean(2)))
    if med > 1e-3:                                        # 1) brightness (gentle gamma)
        img = np.clip(img, 0, 1) ** float(np.clip(np.log(target_median) / np.log(med), 0.82, 1.22))
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128; b = lab[..., 2] - 128; L = lab[..., 0] / 255
    br = L > 0.75
    if br.sum() > 1000:                                  # 2) neutralise the whites
        a = a - float(a[br].mean()) * wb_strength
        b = b - float(b[br].mean()) * wb_strength
    ch = np.sqrt(a * a + b * b); col = ch > 4            # 3) saturation — extremes only
    if col.sum() > 1000:
        cur = float(np.median(ch[col]))
        if cur < 6.0:
            sc = float(np.clip(6.0 / cur, 1.0, 1.25))    # too flat -> mild boost
        elif cur > 16.0:
            sc = float(np.clip(16.0 / cur, 0.80, 1.0))   # over-saturated -> mild tame
        else:
            sc = 1.0                                     # natural -> leave alone
        a = a * sc; b = b * sc
    lab[..., 1] = a + 128; lab[..., 2] = b + 128
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


if AUTO_CALIBRATE:
    print("BEFORE auto-calibrate:", _stats(_base))
    calibrated = auto_finish(_base, TARGET_MEDIAN, WB_STRENGTH)
    print("AFTER  auto-calibrate:", _stats(calibrated))
    cv2.imwrite("hdr_result_final.jpg", calibrated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    plt.figure(figsize=(18, 11))
    plt.imshow(cv2.cvtColor(calibrated, cv2.COLOR_BGR2RGB)); plt.axis("off")
    plt.title("FINAL — auto-calibrated (reference-free)")
    plt.show()
    files.download("hdr_result_final.jpg")
    print("done — hdr_result_final.jpg (consistent AutoHDR-style targets, no reference needed)")
else:
    calibrated = _base
    print("AUTO_CALIBRATE off — using the CELL-5 result as final")


# ============================== CELL 7 — OPTIONAL: BATCH all scenes from Drive =
# Process EVERY sceneN folder under DRIVE_BASE with the CELL-4 chain and write
# each result back to Drive — no per-scene display, just a progress line.
# Requires CELLS 1-4 to have run once (so the finishing functions exist).
import glob
OUT_DIR = "/content/drive/MyDrive/RAW/new-exposure/_autohdr_out"   #@param {type:"string"}
os.makedirs(OUT_DIR, exist_ok=True)

scene_dirs = sorted([d for d in glob.glob(os.path.join(DRIVE_BASE, "*")) if os.path.isdir(d)
                     and not os.path.basename(d).startswith("_")])
print(f"{len(scene_dirs)} scene folders to process -> {OUT_DIR}")

for sd in scene_dirs:
    name = os.path.basename(sd)
    bpaths = sorted(sum([glob.glob(os.path.join(sd, e)) for e in _EXTS], []))
    if len(bpaths) < 2:
        print(f"  {name:10s}: skipped (only {len(bpaths)} image)"); continue
    b = [cv2.imread(p) for p in bpaths]
    b = [im for im in b if im is not None]
    b = [cv2.resize(im, (W, int(im.shape[0] * W / im.shape[1])), interpolation=cv2.INTER_AREA)
         if im.shape[1] != W else im for im in b]
    b.sort(key=lambda im: im.mean())
    al = _align(b)
    mert = (cv2.createMergeMertens().process(al) * 255).clip(0, 255).astype("uint8")
    P, _ = derive_params(mert)                          # per-scene histogram params
    if not AUTO_PARAMS:
        P.update(shadow_fill=SHADOW_FILL, recover=RECOVER_HIGHLIGHTS, expose_tgt=0.68, scurve=0.045)
    fu = _shadow_fill(mert, al[-1], max_fill=P["shadow_fill"])
    fu = _highlight_recover(fu, al[0], max_pull=P["recover"])
    im = fu.astype(np.float32) / 255
    exm = _bright_ramp(im)
    im = _wb(im, exclude=exm); im = _levels_soft(im, exclude=exm)
    im = _expose(im, target=P["expose_tgt"], exclude=exm)
    if "_analysis_dodge_map" in globals():               # analysis-driven local exposure
        im = np.clip(im, 0, 1) ** _analysis_dodge_map(fu)[..., None]
    im = _neutralize(im); im = _match_white(im)
    im = _lift_whites(im, amount=_auto_lift_amount(im), exclude=exm)   # adaptive lift
    im = _scurve(im, P["scurve"]); im = _tame_warm(im); im = _desat_warm(im)
    im = _pinch_warm(im, _auto_warm_amount(im) if AUTO_WARMTH else PINCH_WARMTH)
    im = _sharpen(im)
    im = _soft_highlights(im)                            # keep highlight tone
    res = (im * 255).clip(0, 255).astype("uint8")
    if AUTO_CALIBRATE:                                   # reference-free normalisation
        res = auto_finish(res, TARGET_MEDIAN, WB_STRENGTH)
    outp = os.path.join(OUT_DIR, f"{name}_autohdr.jpg")
    cv2.imwrite(outp, res, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lu = res.astype(np.float32).mean(2) / 255
    print(f"  {name:10s}: {len(al)} brackets -> saved  (blown {100*(lu>0.95).mean():.1f}% "
          f"crushed {100*(lu<0.06).mean():.1f}%)")
print("BATCH done — all results in", OUT_DIR)
