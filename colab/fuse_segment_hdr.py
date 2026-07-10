"""
FUSE -> SEGMENT -> MANUAL VIEW -> HDR   (the clean, simple pipeline)

1. Upload the exposure BRACKETS  -> aligned -> FUSED image (shown).
2. Segment ONLY 4 things from the FUSED image, each with a unique colour:
       WINDOW  (whole unit)       YELLOW outline
       GLASS   (see-through area) CYAN
       FRAME   (bars/mullions)    ORANGE
       CURTAINS (incl. blinds)    MAGENTA
3. YOU upload the darker image yourself — the one where the window background/view
   looks solid and good to YOUR eye (no auto-select).
4. HDR: the room + curtains + frame are enhanced; the GLASS is protected and filled
   with YOUR uploaded image — crisp view, nothing else touched.

Run on Google Colab with a GPU (A100 / T4). Copy each CELL into its own Colab cell.
"""

# ============================== CELL 1 — install ==============================
# (Runtime -> Change runtime type -> GPU, first!)
# !pip install -q -U transformers ultralytics opencv-python-headless
# !pip install -q --force-reinstall "pillow==10.4.0"
#
# >>> AFTER THIS CELL: Runtime -> Restart session, then run CELL 2. <<<


# ============================== CELL 2 — load models ==========================
import os, math, torch, numpy as np, cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import SAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
GD_ID = "IDEA-Research/grounding-dino-base"
gd_proc = AutoProcessor.from_pretrained(GD_ID)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_ID).to(DEVICE).eval()
sam = SAM("sam2.1_l.pt")
print("models loaded")


# ============================== CELL 3 — upload BRACKETS -> fused =============
# Select ALL the exposure brackets of ONE scene (only the brackets, not the view image).
from google.colab import files
up = files.upload()
paths = sorted(up.keys())
imgs = [cv2.imread(p) for p in paths]
imgs = [im for im in imgs if im is not None]
W = 2000
res = []
for im in imgs:
    h, w = im.shape[:2]
    if w != W:
        im = cv2.resize(im, (W, int(h * W / w)), interpolation=cv2.INTER_AREA)
    res.append(im)
res.sort(key=lambda im: im.mean())


def _ecc_align(mov, ref, wwork=1000):
    """Align `mov` to `ref` with an ECC homography (exposure-robust)."""
    s = min(1.0, wwork / ref.shape[1])
    g_r = cv2.equalizeHist(cv2.resize(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None,
                                      fx=s, fy=s, interpolation=cv2.INTER_AREA))
    g_m = cv2.equalizeHist(cv2.resize(cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY), None,
                                      fx=s, fy=s, interpolation=cv2.INTER_AREA))
    warp = np.eye(3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    cv2.findTransformECC(g_r, g_m, warp, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
    S = np.diag([s, s, 1.0])
    Hm = (np.linalg.inv(S) @ warp.astype(np.float64) @ S).astype(np.float32)
    return cv2.warpPerspective(mov, Hm, (ref.shape[1], ref.shape[0]),
                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


REF = res[len(res) // 2]                       # alignment reference (middle bracket)
aligned = []
for i, im in enumerate(res):
    if i == len(res) // 2:
        aligned.append(im); continue
    try:
        aligned.append(_ecc_align(im, REF))
    except cv2.error:
        aligned.append(im)
res = aligned

fused = (cv2.createMergeMertens().process(res) * 255).clip(0, 255).astype("uint8")
cv2.imwrite("fused.jpg", fused)
print(f"{len(res)} brackets (aligned) -> fused {fused.shape[1]}x{fused.shape[0]}")
try:
    from IPython.display import Image as _Img, display
    print("FUSED (this image gets segmented + HDR'd):")
    display(_Img("fused.jpg"))
except Exception:
    pass


# ============================== CELL 4 — helpers ==============================
def detect(bgr, phrase, box_t=0.25, text_t=0.20):
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inp = gd_proc(images=pil, text=phrase + " .", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = gd_model(**inp)
    try:
        d = gd_proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=box_t, text_threshold=text_t,
            target_sizes=[pil.size[::-1]])[0]
    except TypeError:
        d = gd_proc.post_process_grounded_object_detection(
            out, inp.input_ids, box_threshold=box_t, text_threshold=text_t,
            target_sizes=[pil.size[::-1]])[0]
    return d["boxes"].cpu().numpy()


def _iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    ar_a = (a[2] - a[0]) * (a[3] - a[1])
    ar_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(ar_a + ar_b - inter, 1e-6)


def seg_union(bgr, phrases, max_boxes=24):
    """ONE DINO pass for all phrasings (joined prompt) -> IoU-deduped boxes -> SAM.
    Joining the synonyms into a single prompt replaces N model runs with one, and the
    dedup removes the near-identical rectangles the synonyms produce — on big-window
    scenes DINO boxes every pane AND the whole wall, and SAM's runtime scales with the
    number of boxes, so this is a large speedup with identical output."""
    boxes = detect(bgr, " . ".join(phrases))
    if len(boxes) == 0:
        return np.zeros(bgr.shape[:2], bool)
    keep = []
    for b in boxes:
        if all(_iou(b, k) < 0.75 for k in keep):
            keep.append(b)
        if len(keep) >= max_boxes:
            break
    print(f"    ({len(boxes)} boxes -> {len(keep)} after dedup)")
    try:
        r = sam(bgr, bboxes=[list(map(float, b)) for b in keep], retina_masks=True, verbose=False)
    except TypeError:
        r = sam(bgr, bboxes=[list(map(float, b)) for b in keep], verbose=False)
    if not r or r[0].masks is None:
        return np.zeros(bgr.shape[:2], bool)
    return r[0].masks.data.cpu().numpy().astype(bool).any(axis=0)


def tint(base, mask, col, alpha=0.55, contour=True):
    o = base.astype(np.float32)
    a = mask.astype(np.float32)[..., None] * alpha
    t = np.zeros_like(o); t[:] = col
    o = (o * (1 - a) + t * a).clip(0, 255).astype("uint8")
    if contour:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cnts, -1, col, 2)
    return o


def _feather(mask, guide_bgr):
    r = max(6, int(mask.shape[1] * 0.004))
    src = mask.astype(np.float32)
    try:
        gd = cv2.cvtColor((np.clip(guide_bgr, 0, 1) * 255).astype(np.uint8),
                          cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return np.clip(cv2.ximgproc.guidedFilter(gd, src, r, 1e-3), 0, 1)
    except Exception:
        return np.clip(cv2.GaussianBlur(src, (0, 0), r * 0.6), 0, 1)


# ---- HDR finishing (same math as the local hdr.py) ----
def _wb(img, pct=92, exclude=None):
    luma = img.mean(2); valid = luma < 0.98
    if exclude is not None: valid &= exclude < 0.5
    v = luma[valid]
    if v.size < 1000: return img
    ref = img[valid & (luma >= np.percentile(v, pct))]
    if len(ref) < 100: return img
    g = ref.mean(0); g = g.max() / np.maximum(g, 1e-6)
    return np.clip(img * g, 0, 1)

def _levels(img, blk=0.4, wht=99.7, maxblk=0.25, exclude=None):
    luma = img.mean(2); s = luma if exclude is None else luma[exclude < 0.5]
    if s.size < 1000: s = luma
    lo = min(float(np.percentile(s, blk)), maxblk); hi = float(np.percentile(s, wht))
    if hi - lo < 0.1: return img
    return np.clip((img - lo) / (hi - lo), 0, 1) * (0.91 - 0.02) + 0.02

def _expose(img, target=0.70):
    m = max(float(np.median(img.mean(2))), 1e-3)
    return img if m >= target else np.clip(img, 0, 1) ** float(np.clip(np.log(target)/np.log(m), 0.45, 1.0))

def _neutralize(img, strength=1.0):
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]/255; a = lab[..., 1]-128; b = lab[..., 2]-128
    ch = np.sqrt(a*a+b*b)
    w = np.clip((L-0.70)/0.12, 0, 1) * np.clip((22-ch)/8, 0, 1)
    w = np.maximum(w, np.clip((L-0.85)/0.10, 0, 1))
    w = cv2.GaussianBlur(w, (0, 0), 8)*strength
    lab[..., 1] = a*(1-w)+128; lab[..., 2] = b*(1-w)+128
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255

def _match_white(img, strength=0.85):
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    br = lab[..., 0] > 0.78*255
    if br.sum() < 1000: return img
    lab[..., 1] += (128-float(lab[..., 1][br].mean()))*strength
    lab[..., 2] += (128-float(lab[..., 2][br].mean()))*strength
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255

def _scurve(img, s=0.05): return np.clip(img + s*np.sin(2*np.pi*(img-0.5)), 0, 1)

def _sharpen(img, fine=0.9, clarity=0.2, protect=None):
    img = np.clip(img + fine*(img - cv2.GaussianBlur(img, (0, 0), 1.0)), 0, 1)
    wide = cv2.GaussianBlur(img, (0, 0), 15)
    ca = clarity if protect is None else clarity*(1-protect)[..., None]
    return np.clip(img + ca*(img - wide), 0, 1)

print("helpers ready")


# ============================== CELL 5 — SEGMENT the 4 things (from FUSED) =====
H, Wd = fused.shape[:2]

# ---- VERIFICATION setup (light-through physics on a DARK reference bracket) ----
# DINO's labels are proposals, NOT truth (it has called a ceiling fan and bed frames
# "curtain", louvered closet doors "venetian blind" AND "window", a lamp "glass").
# Physics decides: real GLASS shows the bright/colourful exterior in a dark bracket;
# fans, beds, closets, TVs and solid doors are DARK there.
#
# The reference is the bracket whose mean sits near ~0.15 — dark enough that the indoor
# is black, bright enough that the exterior actually READS. Using the absolute darkest
# broke big-window scenes shot down to EV -7 (test-6): at mean ~4/255 even the exterior
# is nearly black, every window failed the test and NOTHING got segmented.
_means = [float(im.mean()) / 255 for im in res]
_vref = int(np.argmin([abs(m - 0.15) for m in _means]))
print(f"verification reference bracket = #{_vref} (mean {_means[_vref]*255:.0f}/255)  [darkest=#0]")
_dk = res[_vref].astype(np.float32) / 255
_dl = _dk.mean(2); _dc = _dk.max(2) - _dk.min(2)
_thr = max(0.22, 0.45 * float(np.percentile(_dl, 99.9)))   # adapts to the reference
print(f"light-through threshold on reference bracket = {_thr:.2f}")


def verify(mask, mode, window_zone=None, min_area=0.0004):
    """Keep only components that pass the physics test.
    mode='window' : must SHOW the exterior (p95 bright OR colourful in darkest).
    mode='curtain': must HANG OVER a verified window AND be mostly dark in darkest
                    (median) — so blinds over glass pass, but fans/beds/closets fail."""
    ncc, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    out = np.zeros(mask.shape, bool)
    for i in range(1, ncc):
        if st[i, cv2.CC_STAT_AREA] < min_area * H * Wd:
            continue
        comp = lab == i
        if mode == "window":
            ok = (float(np.percentile(_dl[comp], 95)) >= _thr or
                  float(np.percentile(_dc[comp], 95)) >= 0.15)
        else:
            over = float((comp & window_zone).sum()) / float(comp.sum())
            ok = over >= 0.30 and float(np.median(_dl[comp])) < _thr * 0.8
        if ok:
            out |= comp
    return out


# 1) WINDOW (whole unit): propose, then VERIFY — drops closet doors, TVs, lamps, solid
#    doors (dark in the darkest bracket = no exterior behind them)
WINDOW_raw = seg_union(fused, ["window", "windowpane", "sliding glass window", "glass door"])
WINDOW = verify(WINDOW_raw, "window")
print(f"window  : proposed {WINDOW_raw.mean()*100:5.1f}% -> verified {WINDOW.mean()*100:5.1f}%")

# 2) CURTAINS: propose, then VERIFY — fans, bed frames and closets don't hang over any
#    verified window, so they are dropped instantly
_zone = cv2.dilate(WINDOW.astype(np.uint8),
                   np.ones((max(3, int(Wd * 0.02) | 1),) * 2, np.uint8)).astype(bool)
CURTAIN_raw = seg_union(fused, ["curtain", "drape", "blind", "venetian blind"])
CURTAIN = verify(CURTAIN_raw, "curtain", window_zone=_zone)
print(f"curtains: proposed {CURTAIN_raw.mean()*100:5.1f}% -> verified {CURTAIN.mean()*100:5.1f}%")

# 3) FRAME: mullions/bars INSIDE the window. THREE conditions, all physics/geometry:
#    a) achromatic (white/grey painted bars, chroma < 0.07 in the fused image)
#    b) NOT see-through — a frame is inside the room, it BLOCKS light, so in the dark
#       reference bracket it stays dark. Blown-out sky / white pool deck / white boats
#       are achromatic too, but they are BRIGHT in the dark bracket (SEE) — never frame.
#       This is what kills the "frame eats the view" holes in the glass mask.
#    c) LINE-shaped — bars survive a horizontal or vertical line opening; leftover
#       irregular patches do not. Big blobs (a whole shaded wall) are subtracted.
f = fused.astype(np.float32) / 255
chroma = f.max(2) - f.min(2)
SEE = (_dl >= _thr) | (_dc >= 0.15)               # see-through in the dark bracket
lip = max(3, int(Wd * 0.004) | 1)
WIN_IN = cv2.erode(WINDOW.astype(np.uint8), np.ones((lip, lip), np.uint8)).astype(bool)
achro = (chroma < 0.07) & WIN_IN & (~SEE)
bar_k = max(9, int(Wd * 0.025) | 1)
blobs = cv2.morphologyEx(achro.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((bar_k, bar_k), np.uint8)).astype(bool)
Lk = max(15, int(Wd * 0.03))
line_h = cv2.morphologyEx(achro.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((1, Lk), np.uint8)).astype(bool)
line_v = cv2.morphologyEx(achro.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((Lk, 1), np.uint8)).astype(bool)
FRAME = (line_h | line_v) & ~blobs

# 4) GLASS = window interior minus frame minus the OPAQUE part of the curtains.
#    A blind/curtain is subtracted ONLY where it actually blocks the light — the
#    see-through gaps between blind slats show the exterior in the darkest bracket,
#    so they STAY glass. Without this, a window fully covered by venetian blinds
#    loses its entire glass mask and the HDR cell has nothing to protect/composite.
CURTAIN_OPQ = CURTAIN & (~SEE)                    # opaque fabric / slats only
GLASS = WIN_IN & ~FRAME & ~CURTAIN_OPQ
GLASS = cv2.morphologyEx(GLASS.astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((5, 5), np.uint8))          # seal hairline seams
GLASS = cv2.morphologyEx(GLASS, cv2.MORPH_OPEN,
                         np.ones((3, 3), np.uint8)).astype(bool)
print(f"glass through curtain gaps kept: curtain {CURTAIN.mean()*100:.1f}% "
      f"-> opaque {CURTAIN_OPQ.mean()*100:.1f}%")

# unique colours (BGR)
COL = {"window": (0, 255, 255),     # YELLOW
       "glass": (255, 255, 0),      # CYAN
       "frame": (0, 140, 255),      # ORANGE
       "curtains": (255, 0, 255)}   # MAGENTA
MASKS = {"window": WINDOW, "glass": GLASS, "frame": FRAME, "curtains": CURTAIN}

os.makedirs("seg", exist_ok=True)
for name, m in MASKS.items():
    print(f"{name:9s}: {m.mean()*100:5.1f}% of image")
    cv2.imwrite(f"seg/overlay_{name}.jpg", tint(fused, m, COL[name]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(f"seg/mask_{name}.png", m.astype(np.uint8) * 255)

# combined: cyan glass + orange frame + magenta curtains, yellow window outline
combo = fused.astype(np.float32)
for name in ("glass", "frame", "curtains"):
    a = MASKS[name].astype(np.float32)[..., None] * 0.55
    t = np.zeros_like(combo); t[:] = COL[name]
    combo = combo * (1 - a) + t * a
combo = combo.clip(0, 255).astype("uint8")
cnts, _ = cv2.findContours(WINDOW.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(combo, cnts, -1, COL["window"], 3)
cv2.imwrite("seg/ALL_segments.jpg", combo, [cv2.IMWRITE_JPEG_QUALITY, 92])

import shutil
shutil.make_archive("segments", "zip", "seg")
try:
    from IPython.display import Image as _Img, display
    print("\nALL — yellow window outline · cyan glass · orange frame · magenta curtains:")
    display(_Img("seg/ALL_segments.jpg"))
    for name in ("window", "glass", "frame", "curtains"):
        print(f"{name}:")
        display(_Img(f"seg/overlay_{name}.jpg"))
except Exception:
    pass
files.download("segments.zip")
print("done — segments.zip has the 4 overlays + 4 binary masks + the combined map")


# ============================== CELL 6 — YOU upload the darker/view image ======
# Upload the ONE image where the window background/view looks solid and good to YOUR
# eye. SELF-CONTAINED: needs only `fused` from CELL 3 — it resizes the upload to the
# fused image's exact size and ECC-aligns it against the fused image directly.
from google.colab import files as _files
print(">>> Upload YOUR darker / view image now:")
vup = _files.upload()
if not vup:
    raise RuntimeError("no file uploaded — run this cell again and pick your view image")
vpath = list(vup.keys())[0]
viewimg = cv2.imread(vpath)
if viewimg is None:
    raise RuntimeError(f"could not read '{vpath}' — is it a JPG/PNG?")
H2, W2 = fused.shape[:2]
viewimg = cv2.resize(viewimg, (W2, H2), interpolation=cv2.INTER_AREA)  # exact same size

# align to the FUSED image (inline ECC homography — exposure-robust)
try:
    s = min(1.0, 1000 / W2)
    g_r = cv2.equalizeHist(cv2.resize(cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY), None,
                                      fx=s, fy=s, interpolation=cv2.INTER_AREA))
    g_m = cv2.equalizeHist(cv2.resize(cv2.cvtColor(viewimg, cv2.COLOR_BGR2GRAY), None,
                                      fx=s, fy=s, interpolation=cv2.INTER_AREA))
    warp = np.eye(3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    cv2.findTransformECC(g_r, g_m, warp, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
    S = np.diag([s, s, 1.0])
    Hm = (np.linalg.inv(S) @ warp.astype(np.float64) @ S).astype(np.float32)
    viewimg = cv2.warpPerspective(viewimg, Hm, (W2, H2),
                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
    print(" * view image aligned to the fused image")
except Exception as e:
    print(f" * WARNING: alignment failed ({e}) — using the image as-is")

VIEW_DESC = f"your image '{vpath}'"
cv2.imwrite("view_image.jpg", viewimg, [cv2.IMWRITE_JPEG_QUALITY, 90])
try:
    from IPython.display import Image as _Img, display
    print("THIS image will fill the glass:")
    display(_Img("view_image.jpg"))
except Exception:
    pass


# ============================== CELL 7 — HDR + composite ======================
# HDR on the room + curtains + frame; the GLASS is protected and then filled with YOUR
# uploaded image (gently lifted so it sits naturally). Nothing else is touched.

# --- RECOVER the OPEN-DOOR / OPEN-WINDOW view -------------------------------------
# An OPEN sliding door is an EMPTY opening — not an object — so DINO/SAM cannot box or
# segment it (they only segmented the glass panels beside it). But the exterior SHOWS
# THROUGH that opening: in YOUR dark view image it is bright/colourful while the indoor
# is dark. So near the detected windows/doors, add those see-through pixels back in.
v = viewimg.astype(np.float32) / 255
vl = v.mean(2); vc = v.max(2) - v.min(2)
zone = cv2.dilate(WINDOW.astype(np.uint8),
                  np.ones((max(3, int(Wd * 0.07) | 1),) * 2, np.uint8)).astype(bool)
see = (vl > 0.45) | (vc > 0.18)
extra = zone & see & (~WINDOW) & (~CURTAIN_OPQ)
extra = cv2.morphologyEx(extra.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((5, 5), np.uint8)).astype(bool)
# keep only pieces ATTACHED to a window — an open-door gap touches the glass panels,
# while a LIT lampshade nearby is bright too but floats on its own -> dropped
_wtouch = cv2.dilate(WINDOW.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
ncc2, lab2 = cv2.connectedComponents(extra.astype(np.uint8))
_keep = np.zeros(extra.shape, bool)
for i in range(1, ncc2):
    comp = lab2 == i
    if (comp & _wtouch).any():
        _keep |= comp
extra = _keep
OPENING = cv2.morphologyEx((WINDOW | extra).astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((max(3, int(Wd * 0.006) | 1),) * 2, np.uint8)).astype(bool)
print(f"open-door view recovered: +{extra.mean()*100:.1f}% of image")

# ---- WHAT to fill with your image -------------------------------------------------
#  FILL_MODE = "glass":  only the see-through panes — the mullions/frame stay BRIGHT
#              from the indoor HDR (pro look; in window mode they get painted dark
#              from your view image). DEFAULT.
#  FILL_MODE = "window": the FULL window unit (glass + blinds + inner frame) is taken
#              from YOUR image — use when blinds/shades are INSIDE the window and your
#              image shows the whole window well-exposed (the test-9 bedroom case).
FILL_MODE = "glass"

# EXCLUDE_CURTAINS: keep the curtains on the HDR side — NEVER filled from your image.
#  True  -> fabric drapes over the window stay bright/enhanced (this scene; without it
#           the drapes were painted DARK from the view image).
#  False -> curtains filled too (venetian blinds INSIDE the window, when your image
#           shows the whole window well-exposed — the test-9 bedroom case).
EXCLUDE_CURTAINS = True

if FILL_MODE == "window":
    FILL = WINDOW | extra
else:
    FILL = (GLASS | extra) & (~CURTAIN_OPQ)
if EXCLUDE_CURTAINS:
    FILL = FILL & (~CURTAIN)
glass_soft = _feather(FILL.astype(np.float32), fused.astype(np.float32) / 255)
glass_soft = glass_soft * OPENING.astype(np.float32)     # never outside the opening
print(f"FILL_MODE={FILL_MODE}: protected = {(glass_soft > 0.5).mean()*100:.1f}% of image (rest gets HDR)")

# indoor HDR (glass excluded from all measurements)
img = fused.astype(np.float32) / 255
ex = (glass_soft > 0.3).astype(np.float32)
img = _wb(img, exclude=ex)
img = _levels(img, exclude=ex)
img = _expose(img, 0.70)
img = _neutralize(img)
img = _match_white(img)
img = _scurve(img, 0.05)
band = np.clip(cv2.GaussianBlur(cv2.dilate((glass_soft > 0.2).astype(np.float32),
               np.ones((max(3, int(Wd*0.01) | 1),) * 2, np.uint8)), (0, 0), 8), 0, 1)
img = _sharpen(img, protect=band)          # clarity damped near the glass edge (no halo)

# composite YOUR image into the glass
print(f"filling the glass with {VIEW_DESC}")
vsrc = viewimg.astype(np.float32) / 255
sel = glass_soft > 0.5
med = float(np.median(vsrc.mean(2)[sel])) if sel.any() else 0.0
if 1e-3 < med < 0.58:                      # gentle lift only if the view sits dark
    vsrc = np.clip(vsrc, 0, 1) ** float(np.clip(np.log(0.58) / np.log(med), 0.6, 1.0))
gv = glass_soft[..., None]
img = img * (1 - gv) + vsrc * gv

result = (img * 255).clip(0, 255).astype("uint8")
cv2.imwrite("hdr_result.jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
prot = tint(fused, glass_soft > 0.5, COL["glass"])
cv2.imwrite("protected_glass.jpg", prot, [cv2.IMWRITE_JPEG_QUALITY, 92])
try:
    from IPython.display import Image as _Img, display
    print("PROTECTED glass (cyan) — HDR applied to everything else:")
    display(_Img("protected_glass.jpg"))
    print("FUSED (base):"); display(_Img("fused.jpg"))
    print(f"FINAL — indoor HDR + glass filled with {VIEW_DESC}:")
    display(_Img("hdr_result.jpg"))
except Exception:
    pass
files.download("protected_glass.jpg")
files.download("hdr_result.jpg")
print("done — room/curtains/frame enhanced; glass = your image, untouched by HDR")
