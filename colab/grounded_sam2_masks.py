"""
Grounded-SAM 2 — PER-CONCEPT precise masks for the hdr-enhance window pull.
Run on Google Colab with a GPU (A100 / T4). Copy each CELL into its own Colab cell.

Pipeline:  Grounding DINO (text -> boxes)  ->  SAM 2.1 (boxes -> pixel-perfect masks)

You upload the exposure BRACKETS; the notebook builds the FUSED image (Mertens) and also
keeps the DARKEST bracket. Then, for EACH concept separately (curtain, blind, window,
frame, glass door, sky, water, tree), it detects + segments and writes its OWN overlay +
binary mask so you can inspect every one in detail and catch mistakes. Structural things
(curtain/frame/blind/window) are read on the well-lit FUSED image; the exterior VIEW
(sky/water/tree) is read on the DARKEST bracket (it blows out in the fusion). Everything
is zipped for download; the .npz feeds the local hdr.py window pull.
"""

# ============================== CELL 1 — install ==============================
# (Runtime -> Change runtime type -> GPU : A100 or T4, first!)
# !pip install -q -U transformers ultralytics supervision opencv-python-headless
# !pip install -q --force-reinstall "pillow==10.4.0"
#
# >>> AFTER THIS CELL: Runtime -> Restart session  (MUST restart, then run CELL 2). <<<


# ============================== CELL 2 — load models ==========================
import os, torch, numpy as np, cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import SAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| gpu:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "-")

GD_ID = "IDEA-Research/grounding-dino-base"
gd_proc = AutoProcessor.from_pretrained(GD_ID)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_ID).to(DEVICE).eval()
sam = SAM("sam2.1_l.pt")                              # SAM 2.1 Hiera-Large (auto-downloads)
print("models loaded")


# ============================== CELL 3 — upload BRACKETS, build fused ==========
# Select ALL the exposure brackets of ONE scene (e.g. the 7-8 test-7 / test4 JPGs).
from google.colab import files
up = files.upload()
paths = sorted(up.keys())
imgs = [cv2.imread(p) for p in paths]
imgs = [im for im in imgs if im is not None]
# resize to a common width, sort darkest -> brightest
W = 2000
res = []
for im in imgs:
    h, w = im.shape[:2]
    if w != W:
        im = cv2.resize(im, (W, int(h * W / w)), interpolation=cv2.INTER_AREA)
    res.append(im)
res.sort(key=lambda im: im.mean())

# ALIGN every bracket to the middle one (ECC homography) BEFORE fusing — without this the
# window pull composites slightly-shifted brackets and the view comes out washed/milky
# (exactly the washed left window you saw). This is what the local pipeline does.
def _align(imgs, wwork=1000):
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
            H = (np.linalg.inv(S) @ warp.astype(np.float64) @ S).astype(np.float32)
            out.append(cv2.warpPerspective(im, H, (w, h),
                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE))
        except cv2.error:
            out.append(im)
    return out

res = _align(res)
darkest = res[0]
fused = (cv2.createMergeMertens().process(res) * 255).clip(0, 255).astype("uint8")
cv2.imwrite("fused.jpg", fused)
print(f"{len(res)} brackets (aligned) -> fused {fused.shape[1]}x{fused.shape[0]}; darkest mean={res[0].mean():.0f}")


# ============================== CELL 4 — helpers ==============================
def detect(bgr, phrase, box_t=0.25, text_t=0.20):
    """Grounding DINO: return boxes (xyxy) for a single concept phrase on a BGR image."""
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
    return d["boxes"].cpu().numpy(), d["scores"].cpu().numpy()


def segment(bgr, boxes):
    """SAM 2.1: union of pixel-precise masks for the given boxes."""
    if len(boxes) == 0:
        return np.zeros(bgr.shape[:2], bool)
    r = sam(bgr, bboxes=boxes.tolist(), verbose=False)
    if not r or r[0].masks is None:
        return np.zeros(bgr.shape[:2], bool)
    return r[0].masks.data.cpu().numpy().astype(bool).any(axis=0)


def overlay(base_bgr, mask, colour):
    o = base_bgr.astype(np.float32)
    a = mask[..., None] * 0.5
    o = o * (1 - a) + np.array(colour, np.float32) * a
    # draw a crisp contour so the boundary is clearly visible
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    o = o.clip(0, 255).astype("uint8")
    cv2.drawContours(o, cnts, -1, colour, 2)
    return o


# ============================== CELL 5 — per-concept detect + segment =========
# Structural concepts read on the FUSED image; VIEW concepts on the DARKEST bracket.
STRUCT = ["curtain", "blind", "window", "window frame", "glass door", "wall", "floor"]
VIEW = ["sky", "water", "tree", "umbrella"]     # umbrella is OUTDOOR view, not a curtain
COLOURS = {"curtain": (200, 0, 200), "blind": (200, 0, 120), "window": (200, 200, 0),
           "window frame": (0, 140, 255), "glass door": (0, 90, 255), "wall": (120, 120, 120),
           "floor": (60, 90, 160), "sky": (255, 150, 0), "water": (255, 255, 0),
           "tree": (0, 200, 0), "umbrella": (0, 200, 200)}

# see-through EXTERIOR mask (from the darkest bracket): bright OR colourful = outdoors.
# Used to strip exterior objects (a closed pool umbrella) that get mislabelled 'curtain'.
_d = darkest.astype(np.float32) / 255
_dl = _d.mean(2); _dc = _d.max(2) - _d.min(2)
SEETHRU = np.maximum(np.clip((_dl - 0.06) / 0.10, 0, 1),
                     np.clip((_dc - 0.05) / 0.08, 0, 1)) > 0.4

os.makedirs("out", exist_ok=True)
concept_masks = {}
print("=== per-concept detection ===")
for name in STRUCT + VIEW:
    src = darkest if name in VIEW else fused
    boxes, scores = detect(src, name)
    m = segment(src, boxes)
    # curtains/blinds are OPAQUE indoor fabric — drop any part that is really the
    # see-through exterior (fixes the pool umbrella being called a curtain).
    if name in ("curtain", "blind"):
        m = m & (~SEETHRU)
    concept_masks[name] = m
    ov = overlay(fused, m, COLOURS.get(name, (255, 255, 255)))   # always show on fused
    cv2.imwrite(f"out/overlay_{name.replace(' ', '_')}.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(f"out/mask_{name.replace(' ', '_')}.png", (m.astype(np.uint8) * 255))
    sc = f"{scores.max():.2f}" if len(scores) else "-"
    print(f"  {name:14s} on {'darkest' if name in VIEW else 'fused ':7s}: "
          f"{len(boxes):2d} boxes, area={m.mean()*100:5.1f}%, top_score={sc}")


# ============================== CELL 6 — combined map + zip + download =========
combo = fused.astype(np.float32)
for name, m in concept_masks.items():
    a = m[..., None] * 0.45
    combo = combo * (1 - a) + np.array(COLOURS.get(name, (255, 255, 255)), np.float32) * a
cv2.imwrite("out/ALL_concepts.jpg", combo.clip(0, 255).astype("uint8"), [cv2.IMWRITE_JPEG_QUALITY, 92])

# one npz with every concept mask (uint8 0/255) for the local hdr.py pull
np.savez_compressed("out/gsam2_masks.npz",
                    **{k.replace(" ", "_"): (v.astype(np.uint8) * 255)
                       for k, v in concept_masks.items()})

import shutil
shutil.make_archive("gsam2_results", "zip", "out")
try:
    from IPython.display import Image as _Img, display
    print("\nALL concepts:"); display(_Img("out/ALL_concepts.jpg"))
    for name in STRUCT + VIEW:
        p = f"out/overlay_{name.replace(' ', '_')}.jpg"
        print(name + ":"); display(_Img(p))
except Exception:
    pass
files.download("gsam2_results.zip")
print("done — gsam2_results.zip has per-concept overlays + masks + the combined map + npz")


# ============================== CELL 7 — HDR finishing + window-pull helpers ===
# Same math as the local hdr.py, but the masks come from Grounded-SAM (not SegFormer).
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

def _neutralize(img, boost=None, protect=None, strength=1.0):
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]/255; a = lab[..., 1]-128; b = lab[..., 2]-128
    ch = np.sqrt(a*a+b*b)
    w = np.clip((L-0.70)/0.12, 0, 1) * np.clip((22-ch)/8, 0, 1)
    if boost is not None:
        w = np.maximum(w, boost*np.clip((L-0.55)/0.20, 0, 1)*np.clip((35-ch)/10, 0, 1))
    if protect is not None: w = w*(1-protect)
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

def _feather(mask, guide_bgr):
    r = max(6, int(mask.shape[1]*0.004))
    try:
        gd = cv2.cvtColor((np.clip(guide_bgr, 0, 1)*255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        return np.clip(cv2.ximgproc.guidedFilter(gd, mask.astype(np.float32), r, 1e-3), 0, 1)
    except Exception:
        return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), r*0.6), 0, 1)

def _pick_view(brackets, seg):
    reg = seg > 0.3; nreg = int(reg.sum())
    if nreg < 50: return 0
    best, bi = -1, 0
    for i, im in enumerate(brackets):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
        good = reg & (g > 0.06) & (g < 0.90)
        if int(good.sum()) < 50: continue
        sc = float(np.sqrt(gx*gx+gy*gy)[good].sum())/nreg
        if sc > best: best, bi = sc, i
    return bi

def _pick_frame(brackets, seg, target=0.62):
    d = brackets[0].astype(np.float32)/255; dl = d.mean(2); dc = d.max(2)-d.min(2)
    fr = (seg > 0.3) & (dl < 0.18) & (dc < 0.06)
    if int(fr.sum()) < 50: return len(brackets)-1
    best_i, best_e = -1, 1e9
    for i, im in enumerate(brackets):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        fb = float(np.median(g[fr]))
        if fb < 0.30: continue
        if abs(fb-target) < best_e: best_e, best_i = abs(fb-target), i
    return best_i if best_i >= 0 else len(brackets)-1
print("helpers ready")


# ============================== CELL 8 — run enhance + window pull =============
H, Wd = fused.shape[:2]
zeros = np.zeros((H, Wd), bool)

# window = window (+ glass door) from SAM; surround = curtain (+ blind)
win = concept_masks.get("window", zeros).astype(np.float32)
if concept_masks.get("glass door") is not None:
    win = np.clip(win + concept_masks["glass door"], 0, 1)
curt = concept_masks.get("curtain", zeros).astype(np.float32)
if concept_masks.get("blind") is not None:
    curt = np.clip(curt + concept_masks["blind"], 0, 1)

# subtract surround from window ONLY where the darkest bracket is NOT see-through
d = darkest.astype(np.float32)/255; dl = d.mean(2); dc = d.max(2)-d.min(2)
view_dark = np.maximum(np.clip((dl-0.06)/0.10, 0, 1), np.clip((dc-0.05)/0.08, 0, 1))
k = max(3, int(Wd*0.004) | 1)
cd = cv2.dilate(curt, np.ones((k, k), np.uint8))
seg_win = np.clip(win - cd*(view_dark < 0.4).astype(np.float32), 0, 1)
have_win = float(seg_win.max()) > 1e-3

img = fused.astype(np.float32)/255
# interior finishing — window excluded from WB / levels measurement
wex = cv2.GaussianBlur(cv2.dilate((seg_win > 0.3).astype(np.float32),
                                  np.ones((k, k), np.uint8)), (0, 0), max(2.0, Wd*0.0015))
boost = concept_masks.get("wall")
protect = concept_masks.get("floor")
img = _wb(img, exclude=wex)
img = _levels(img, exclude=wex)
img = _expose(img, 0.70)
img = _neutralize(img, boost=None if boost is None else boost.astype(np.float32),
                  protect=None if protect is None else protect.astype(np.float32))
img = _match_white(img)
img = _scurve(img, 0.05)

# WINDOW PULL — glass from the clearest bracket, frame from a lit bracket, single-source
if have_win:
    vi = _pick_view(res, seg_win); fi = _pick_frame(res, seg_win)
    print(f"window: glass bracket #{vi}, frame bracket #{fi} of {len(res)}")
    raw = res[vi].astype(np.float32)/255; frame_src = res[fi].astype(np.float32)/255
    vlum = raw.mean(2); vchr = raw.max(2)-raw.min(2)
    litw = np.maximum(np.clip((vlum-0.06)/0.10, 0, 1), np.clip((vchr-0.05)/0.08, 0, 1))
    sel = (seg_win > 0.3) & (litw > 0.4)
    med = float(np.median(vlum[sel])) if sel.any() else 0.0
    view = raw if not (1e-3 < med < 0.58) else np.clip(raw, 0, 1)**float(np.clip(np.log(0.58)/np.log(med), 0.6, 1.0))
    winb = (seg_win > 0.25).astype(np.float32)
    winb = cv2.morphologyEx(winb, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    gm = _feather(winb*litw, raw)[..., None]
    unit = frame_src*(1-gm) + view*gm
    wm = np.clip(cv2.GaussianBlur(cv2.dilate(winb, np.ones((3, 3), np.uint8)),
                                  (0, 0), max(2.0, Wd*0.0015)), 0, 1)[..., None]
    img = img*(1-wm) + unit*wm

# sharpen (clarity damped around the window so the boundary can't halo)
band = None
if have_win:
    kk = max(3, int(Wd*0.01) | 1)
    band = np.clip(cv2.GaussianBlur(cv2.dilate((seg_win > 0.2).astype(np.float32),
                                               np.ones((kk, kk), np.uint8)), (0, 0), kk*0.5), 0, 1)
img = _sharpen(img, protect=band)

result = (img*255).clip(0, 255).astype("uint8")
cv2.imwrite("hdr_result.jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
try:
    from IPython.display import Image as _Img, display
    print("FINAL HDR:"); display(_Img("hdr_result.jpg"))
except Exception:
    pass
files.download("hdr_result.jpg")
print("done — final enhanced HDR with Grounded-SAM window pull")
