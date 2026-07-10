"""
WINDOW GLASS FROM THE DARKER IMAGE — v2: SegFormer "windowpane" + light physics

THE model that EXACTLY answers "where does the outdoor view come from":
SegFormer-b5 (ADE20K) has a real per-pixel "windowpane" class — the see-through
glass INCLUDING everything visible behind it. It inherently excludes curtains,
blinds, lamps and TVs (they get their own classes). No DINO, no SAM, no boxes.

CELL 5 chain (semantic mask + the physics proven in v1):
  0) SegFormer looks at a gamma-LIFTED copy (a dark bracket hides windows from any
     model); ALL physics still runs on the ORIGINAL dark image.
  1) GLASS0 = "windowpane" pixels (+ "door" pixels only where light shows through).
  2) frame BARS removed: near-black + colourless + thin LINE-shaped in the dark
     image; THICK dark patches inside mostly-bright glass are the view in shade
     (soffit corner) and are released.
  3) GROW into adjacent see-through pixels (snaps the blocky semantic edges out to
     the true pane edge; stops at the dark frame by itself).
  4) TRIM thin glow leaks; 4b) SQUARE UP each pane to its convex hull (+15% guard).

Verified against the user's white-marked ground truth on P-5 / P-6 / P-7 and
regression-checked on test-9/10/11 (blinds stay out; lamp is 'lamp', never glass).

Run on Google Colab with a GPU. Copy each CELL into its own Colab cell.
"""

# ============================== CELL 1 — install ==============================
# (Runtime -> Change runtime type -> GPU, first!)
# !pip install -q -U transformers opencv-python-headless
# !pip install -q --force-reinstall "pillow==10.4.0"
#
# >>> AFTER THIS CELL: Runtime -> Restart session, then run CELL 2. <<<


# ============================== CELL 2 — load the model =======================
import os, torch, numpy as np, cv2
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
SEG_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
seg_proc = AutoImageProcessor.from_pretrained(SEG_ID)
seg_model = SegformerForSemanticSegmentation.from_pretrained(SEG_ID).to(DEVICE).eval()
print("SegFormer loaded")


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
    print("THIS image gets segmented:")
    display(_Img("dark_input.jpg"))
except Exception:
    pass


# ============================== CELL 4 — helpers ==============================
def label_map(bgr):
    """SegFormer ADE20K per-pixel class map at full image resolution."""
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inp = seg_proc(images=pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = seg_model(**inp)
    lg = torch.nn.functional.interpolate(out.logits, size=bgr.shape[:2],
                                         mode="bilinear", align_corners=False)
    return lg.argmax(1)[0].cpu().numpy()


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


# ============================== CELL 5 — WINDOW GLASS =========================
H, Wd = dark.shape[:2]

# 0) the model must SEE: a dark bracket hides the dimmer windows — let SegFormer
#    look at a gamma-lifted copy; the physics stays on the ORIGINAL dark image.
det = (np.clip((dark.astype(np.float32) / 255) ** 0.5, 0, 1) * 255).astype("uint8")

# light physics of the ORIGINAL dark image
d = dark.astype(np.float32) / 255
dl = d.mean(2); dc = d.max(2) - d.min(2)
thr = max(0.22, 0.45 * float(np.percentile(dl, 99.9)))
see = (dl >= thr) | (dc >= 0.15)                 # the view: bright OR colourful

# 1) semantic glass: 'windowpane' (8) everywhere; 'door' (14) only where light
#    actually shows through it (a glass door passes, a solid wood door is dark)
print("SegFormer: windowpane ...")
lm = label_map(det)
GLASS = (lm == 8) | ((lm == 14) & see)
print(f"semantic glass: {GLASS.mean()*100:.1f}% of image")
# drop specks — a real pane is never smaller than 0.05% of the image
ncc, lab, st, _ = cv2.connectedComponentsWithStats(GLASS.astype(np.uint8))
for i in range(1, ncc):
    if st[i, cv2.CC_STAT_AREA] < 0.0005 * H * Wd:
        GLASS[lab == i] = False

# 2) remove frame bars: near-black + colourless + thin LINE-shaped in the dark image
bar_cand = GLASS & (~see) & (dc < 0.07)
Lk = max(15, int(Wd * 0.03))
line_h = cv2.morphologyEx(bar_cand.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((1, Lk), np.uint8)).astype(bool)
line_v = cv2.morphologyEx(bar_cand.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((Lk, 1), np.uint8)).astype(bool)
BARS = line_h | line_v
# thick dark patches inside MOSTLY-BRIGHT glass = the view in shade -> release
tk = max(9, int(Wd * 0.03) | 1)
_thick = cv2.morphologyEx(bar_cand.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((tk, tk), np.uint8))
_thick = cv2.dilate(_thick, np.ones((max(3, tk // 4),) * 2, np.uint8)).astype(bool)
_clear = np.zeros((H, Wd), bool)
ncc2, lab2 = cv2.connectedComponents(GLASS.astype(np.uint8))
for i in range(1, ncc2):
    comp = lab2 == i
    if see[comp].mean() >= 0.45:
        _clear |= cv2.dilate(comp.astype(np.uint8),
                             np.ones((tk, tk), np.uint8)).astype(bool)
BARS = BARS & ~(_thick & _clear)
GLASS = GLASS & ~BARS
GLASS = cv2.morphologyEx(GLASS.astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((5, 5), np.uint8))
GLASS = cv2.morphologyEx(GLASS, cv2.MORPH_OPEN,
                         np.ones((5, 5), np.uint8)).astype(bool)
print(f"after bars removed: glass {GLASS.mean()*100:.1f}% of image")

# 3) grow into adjacent see-through pixels — snaps the blocky semantic edges out
#    to the true pane edge; the dark frame stops the growth by itself
allowed = (GLASS | see).astype(np.uint8)
grow = GLASS.astype(np.uint8)
_k3 = np.ones((3, 3), np.uint8)
for _ in range(max(10, int(Wd * 0.02))):
    grow = cv2.dilate(grow, _k3) & allowed
GLASS = grow.astype(bool)
print(f"after edge growth: glass {GLASS.mean()*100:.1f}% of image")

# 4) trim thin glow leaks past the pane edge
mm = max(9, int(Wd * 0.015) | 1)
GLASS = cv2.morphologyEx(GLASS.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((mm, mm), np.uint8)).astype(bool)
print(f"after trim of thin leaks: glass {GLASS.mean()*100:.1f}% of image")

# 4b) square up: fill dim pane corners by completing each piece to its convex hull
#     (max +15% area — occluded/ragged pieces stay untouched); bars re-subtracted
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

# 5) POLISH to a clean BLACK/WHITE mask — solid white panes, clean straight edges:
#    a) fill enclosed holes (a pane is solid; bars stay black — they touch the frame)
#    b) simplify each contour to a clean polygon (kills the blocky staircase edges)
ff = 255 - GLASS.astype(np.uint8) * 255
_seed = None
for x in range(0, Wd, 50):
    if not GLASS[0, x]:
        _seed = (x, 0); break
if _seed is not None:
    ffm = np.zeros((H + 2, Wd + 2), np.uint8)
    cv2.floodFill(ff, ffm, _seed, 0)
    GLASS = GLASS | (ff > 0)                     # what floodfill missed = holes
cnts, _ = cv2.findContours(GLASS.astype(np.uint8), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
_poly = np.zeros((H, Wd), np.uint8)
for c in cnts:
    eps = 0.004 * cv2.arcLength(c, True)
    cv2.fillPoly(_poly, [cv2.approxPolyDP(c, eps, True)], 1)
GLASS = _poly.astype(bool) & ~BARS
#    c) re-fill scratches: a thin dark palm trunk / sign edge inside the view leaves
#       a hairline scratch fully ENCLOSED in white -> fill it. A real rail/mullion
#       touches the black frame at both ends -> stays black.
GLASS = cv2.morphologyEx(GLASS.astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((3, 3), np.uint8)).astype(bool)
ff = 255 - GLASS.astype(np.uint8) * 255
_seed = None
for x in range(0, Wd, 50):
    if not GLASS[0, x]:
        _seed = (x, 0); break
if _seed is not None:
    ffm = np.zeros((H + 2, Wd + 2), np.uint8)
    cv2.floodFill(ff, ffm, _seed, 0)
    GLASS = GLASS | (ff > 0)
print(f"after polish (holes filled, edges straightened): glass {GLASS.mean()*100:.1f}% of image")

# yellow = the FINAL pane outlines (the real 2D glass shapes, where the view comes)
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
    print("\nBLACK/WHITE MASK — white = where the outdoor view comes:")
    display(_Img("mask_glass.png"))
    print("\nPANE OUTLINES (yellow) on your image:")
    display(_Img("panes_debug.jpg"))
    print("\nWINDOW GLASS (cyan):")
    display(_Img("overlay_glass.jpg"))
except Exception:
    pass
files.download("mask_glass.png")
files.download("panes_debug.jpg")
files.download("overlay_glass.jpg")
print("done — mask_glass.png (black/white) + panes_debug.jpg + overlay_glass.jpg")
