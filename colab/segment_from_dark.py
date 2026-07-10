"""
WINDOW GLASS FROM THE DARKER IMAGE  (Grounding DINO pane boxes + light physics)

You upload ONE darker image (the one where the window view is visible) and get ONE
thing: the FULL WINDOW GLASS — everything the view shows through each pane (CYAN),
plus a yellow outline of every pane. No frame, no curtains, no HDR.

How CELL 5 works (each step exists because a real defect demanded it):
  1) DINO pane BOXES = the glass (filled). NO SAM here — SAM outlines ONE object
     per box, so a pane of palms+chairs+pool comes back with HOLES; a glass pane
     is not an object, it is a view of many things.
  2) frame BARS removed: near-black + colourless + LINE-shaped in the dark image.
     THICK dark patches inside a mostly-bright pane are released (view in shade,
     e.g. soffit corner) — but stay removed in mostly-dark panes (closed blinds).
  3) GROW into see-through edges the straight boxes missed (tilted panes).
  4) TRIM thin glow leaks past the pane edge; 4b) SQUARE UP each pane to its
     convex hull (fills dim corners; +15% area guard protects blind scenes).
  5) lit OBSTACLES (lamp/TV) detected by DINO and subtracted (guarded so a bright
     window can never be "a TV").

Verified on: test-6/P-5/P-6 big pool window (perfect, matches the user's marked
panes) and test-9/10/11 blinds bedrooms (blinds correctly stay out of the glass).

Run on Google Colab with a GPU. Copy each CELL into its own Colab cell.
CELLs 1, 2 and 4 are IDENTICAL to fuse_segment_hdr.py — if that session is already
running with models loaded, you only need CELL 3 and CELL 5.
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


# ============================== CELL 3 — upload the DARKER image ==============
# Upload ONE image: the darker exposure where the window view/background is visible.
from google.colab import files
up = files.upload()
if not up:
    raise RuntimeError("no file uploaded — run this cell again and pick the darker image")
_name = list(up.keys())[0]
dark = cv2.imread(_name)
if dark is None:
    raise RuntimeError(f"could not read '{_name}' — upload a JPG/PNG")
W = 2000
h, w = dark.shape[:2]
if w != W:
    dark = cv2.resize(dark, (W, int(h * W / w)), interpolation=cv2.INTER_AREA)
cv2.imwrite("dark_input.jpg", dark)
print(f"darker image: {_name} -> {dark.shape[1]}x{dark.shape[0]} (mean {dark.mean():.0f}/255)")
try:
    from IPython.display import Image as _Img, display
    print("THIS image gets segmented by the trained models:")
    display(_Img("dark_input.jpg"))
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
    """ONE DINO pass for all phrasings (joined prompt) -> IoU-deduped boxes -> SAM."""
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


# ============================== CELL 5 — WINDOW GLASS from the darker image ===
# ONE job: get the FULL window glass (everything the view shows through each pane)
# from YOUR darker image.
#
# WHY NO SAM HERE: SAM outlines ONE object per box. A pane of sky = one object ->
# full mask. A pane of palms + chairs + pool = MANY objects -> SAM keeps some and
# leaves HOLES (verified on the test-6/P-5 dark image). A glass pane is not an
# object — it is a view of many things — so:
#   1) Grounding DINO's pane BOX defines the glass area (box = pane, filled fully;
#      container boxes holding 2+ other boxes are the whole-window box -> dropped)
#   2) only the FRAME BARS inside are removed — they are near-BLACK, colourless,
#      LINE-shaped in the dark image (the view is bright/colourful there, never cut)
H, Wd = dark.shape[:2]

# The DETECTOR must SEE: on a very dark bracket DINO goes half-blind — it only boxes
# the sky-bright parts and misses whole panes in shade (verified on P-7, mean 17/255).
# So DINO looks at a gamma-LIFTED copy; ALL the light physics (see / bars) still runs
# on the ORIGINAL dark image, where only true exterior light is bright.
det = (np.clip((dark.astype(np.float32) / 255) ** 0.5, 0, 1) * 255).astype("uint8")

print("model: window glass pane boxes ...")
boxes = detect(det, "window glass . glass pane . windowpane . sliding glass window . glass door")
print(f"    {len(boxes)} raw boxes")
keep = []
for b in boxes:
    if all(_iou(b, k) < 0.75 for k in keep):
        keep.append(b)


def _contains(B, b):
    x0 = max(B[0], b[0]); y0 = max(B[1], b[1])
    x1 = min(B[2], b[2]); y1 = min(B[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ab = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    return inter / ab > 0.85


panes = [B for B in keep if sum(_contains(B, b) for b in keep if b is not B) < 2]
print(f"    {len(keep)} after dedup -> {len(panes)} pane boxes (container boxes dropped)")
if not panes:
    raise RuntimeError("no pane boxes found — try a slightly brighter dark image")

# 1) glass area = the pane boxes, FILLED (tiny inset keeps the seam off the casing)
FILLED = np.zeros((H, Wd), bool)
for b in panes:
    x0, y0, x1, y1 = [int(v) for v in b]
    ix, iy = int((x1 - x0) * 0.015), int((y1 - y0) * 0.015)
    FILLED[y0 + iy:y1 - iy, x0 + ix:x1 - ix] = True

# 2) remove ONLY the frame bars: near-black + colourless + straight LINE shaped
d = dark.astype(np.float32) / 255
dl = d.mean(2); dc = d.max(2) - d.min(2)
thr = max(0.22, 0.45 * float(np.percentile(dl, 99.9)))
see = (dl >= thr) | (dc >= 0.15)                 # the view: bright OR colourful
bar_cand = FILLED & (~see) & (dc < 0.07)
Lk = max(15, int(Wd * 0.03))
line_h = cv2.morphologyEx(bar_cand.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((1, Lk), np.uint8)).astype(bool)
line_v = cv2.morphologyEx(bar_cand.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((Lk, 1), np.uint8)).astype(bool)
BARS = line_h | line_v
# a real bar is THIN — a rail/mullion is only ~1-2% of the image width. A THICK dark
# patch inside a MOSTLY-BRIGHT pane is the VIEW in deep shade (soffit corner behind
# the curtain), never a bar — release it, or the pane corner gets eaten forever.
# But inside a MOSTLY-DARK pane (closed venetian blinds) a thick dark patch IS the
# blinds — keep it removed, else the whole blind becomes "glass".
tk = max(9, int(Wd * 0.03) | 1)
_thick = cv2.morphologyEx(bar_cand.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((tk, tk), np.uint8))
_thick = cv2.dilate(_thick, np.ones((max(3, tk // 4),) * 2, np.uint8)).astype(bool)
_clear = np.zeros((H, Wd), bool)                 # panes that are mostly see-through
for b in panes:
    x0, y0, x1, y1 = [int(v) for v in b]
    if see[max(0, y0):y1, max(0, x0):x1].mean() >= 0.45:
        _clear[max(0, y0):y1, max(0, x0):x1] = True
BARS = BARS & ~(_thick & _clear)
GLASS = FILLED & ~BARS
GLASS = cv2.morphologyEx(GLASS.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((5, 5), np.uint8)).astype(bool)
print(f"filled {FILLED.mean()*100:.1f}% -> bars {BARS.mean()*100:.1f}% "
      f"-> glass {GLASS.mean()*100:.1f}% of image")

# 3) grow into adjacent see-through pixels the straight boxes missed — the window
#    is photographed at an angle, so pane edges are TILTED and a straight rectangle
#    cuts small corners/strips. Those missed bits are bright/colourful (the view),
#    the frame/curtain beside them is dark, so the growth fills exactly the wedges
#    and stops at the frame by itself.
allowed = (GLASS | see).astype(np.uint8)
grow = GLASS.astype(np.uint8)
_k3 = np.ones((3, 3), np.uint8)
for _ in range(max(10, int(Wd * 0.03))):
    grow = cv2.dilate(grow, _k3) & allowed
GLASS = grow.astype(bool)
print(f"after growth into see-through edges: glass {GLASS.mean()*100:.1f}% of image")

# 4) trim thin leaks: a very bright view (white soffit, pool water) puts a faint
#    GLOW on the frame edge next to it in the dark image; those glow pixels pass the
#    see test and the growth walks over them, sticking out past the pane edge. A pane
#    is a big solid shape — any protrusion thinner than ~1.5% of the width is a leak.
mm = max(9, int(Wd * 0.015) | 1)
GLASS = cv2.morphologyEx(GLASS.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((mm, mm), np.uint8)).astype(bool)
print(f"after trim of thin leaks: glass {GLASS.mean()*100:.1f}% of image")

# 4b) square up each pane: a pane is a CONVEX QUAD, but its corner can sit in deep
#     shade (soffit behind the curtain edge) where the growth cannot reach — fill
#     each pane piece to its convex hull. Guards: the hull may add at most 15% area
#     (ragged blind-gap pieces stay untouched), and the frame BARS are re-subtracted
#     so the rail/mullion can never be swallowed.
ncc5, lab5 = cv2.connectedComponents(GLASS.astype(np.uint8))
_sq = np.zeros((H, Wd), np.uint8)
for i in range(1, ncc5):
    comp = (lab5 == i).astype(np.uint8)
    pts = cv2.findNonZero(comp)
    hull = cv2.convexHull(pts)
    filled = np.zeros((H, Wd), np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    _sq |= filled if int(filled.sum()) <= 1.15 * int(comp.sum()) else comp
GLASS = _sq.astype(bool) & ~BARS
print(f"after corner square-up: glass {GLASS.mean()*100:.1f}% of image")

# 5) subtract lit OBSTACLES in front of the glass — a glowing lamp or a TV inside a
#    window box is bright but it is NOT the view. The models detect them; a guard
#    skips any "obstacle" box that is really a whole pane (IoU>0.5 with a pane box),
#    so a bright window can never be mistaken for a TV and deleted.
ob = detect(det, "lamp . lampshade . table lamp . television . tv screen",
            box_t=0.30, text_t=0.25)
OB = np.zeros((H, Wd), bool)
n_ob = 0
for b in ob:
    if any(_iou(b, p) > 0.5 for p in panes):
        continue
    x0, y0, x1, y1 = [int(v) for v in b]
    OB[max(0, y0):y1, max(0, x0):x1] = True
    n_ob += 1
if OB.any():
    cut = float((GLASS & OB).mean())
    GLASS = GLASS & ~OB
    print(f"obstacles removed (lamp/tv, {n_ob} boxes): -{cut*100:.2f}% of image")

# yellow = the FINAL pane outlines (the real 2D glass shapes, where the view comes)
# — NOT DINO's raw straight boxes, which look wrong on an angled window
dbg = dark.copy()
cnts, _ = cv2.findContours(GLASS.astype(np.uint8), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(dbg, cnts, -1, (0, 255, 255), 3)
cv2.imwrite("panes_debug.jpg", dbg, [cv2.IMWRITE_JPEG_QUALITY, 92])

CYAN = (255, 255, 0)
cv2.imwrite("overlay_glass.jpg", tint(dark, GLASS, CYAN),
            [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite("mask_glass.png", GLASS.astype(np.uint8) * 255)

try:
    from IPython.display import Image as _Img, display
    print("\nPANE OUTLINES (yellow) — the final glass shapes, where the view comes:")
    display(_Img("panes_debug.jpg"))
    print("\nWINDOW GLASS (cyan) — full panes, frame bars removed:")
    display(_Img("overlay_glass.jpg"))
except Exception:
    pass
files.download("panes_debug.jpg")
files.download("overlay_glass.jpg")
files.download("mask_glass.png")
print("done — panes_debug.jpg + overlay_glass.jpg + mask_glass.png")
