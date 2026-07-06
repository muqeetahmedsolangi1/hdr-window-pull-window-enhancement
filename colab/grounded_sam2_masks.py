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
import math
up = files.upload()
paths = sorted(up.keys())


def _exptime(path):
    """EXIF exposure time (seconds) — used to label each bracket with its EV."""
    try:
        from PIL import Image as PImage, ExifTags
        ex = PImage.open(path)._getexif() or {}
        tags = {v: k for k, v in ExifTags.TAGS.items()}
        t = ex.get(tags.get("ExposureTime"))
        return float(t) if t else None
    except Exception:
        return None


# resize to a common width, keep the file/EXIF with each image, sort darkest -> brightest
W = 2000
items = []
for p in paths:
    im = cv2.imread(p)
    if im is None:
        continue
    h, w = im.shape[:2]
    if w != W:
        im = cv2.resize(im, (W, int(h * W / w)), interpolation=cv2.INTER_AREA)
    items.append({"img": im, "name": p, "t": _exptime(p)})
items.sort(key=lambda it: it["img"].mean())
res = [it["img"] for it in items]

# EV of each bracket, relative to the median exposure time (same as the local app)
times = [it["t"] for it in items if it["t"]]
med_t = sorted(times)[len(times) // 2] if times else None
EVS = []
for it in items:
    ev = round(math.log2(it["t"] / med_t), 1) if (it["t"] and med_t) else None
    it["ev"] = ev
    EVS.append(ev)

# ---- SHOW every input bracket with its EV (darkest -> brightest) ----
tiles = []
th_h = int(res[0].shape[0] * 330 / W)
for i, it in enumerate(items):
    th = cv2.resize(it["img"], (330, th_h))
    ev = f"EV {it['ev']:+g}" if it["ev"] is not None else "EV ?"
    role = " DARKEST" if i == 0 else ""
    cv2.rectangle(th, (0, 0), (329, 30), (0, 0, 0), -1)
    cv2.putText(th, f"#{i} {ev}{role}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0) if i == 0 else (255, 255, 255), 2)
    tiles.append(th)
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
if len(rows) > 1 and rows[-1].shape[1] != rows[0].shape[1]:
    pad = np.zeros((rows[-1].shape[0], rows[0].shape[1] - rows[-1].shape[1], 3), np.uint8)
    rows[-1] = np.hstack([rows[-1], pad])
cv2.imwrite("inputs_ev.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("input brackets (darkest -> brightest):",
      [f"#{i} EV{e:+g}" if e is not None else f"#{i} ?" for i, e in enumerate(EVS)])
try:
    from IPython.display import Image as _Img, display
    display(_Img("inputs_ev.jpg"))
except Exception:
    pass

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

# show the darker image kept for the fusion gates + the fused base
cv2.imwrite("darkest.jpg", darkest, [cv2.IMWRITE_JPEG_QUALITY, 90])
try:
    from IPython.display import Image as _Img, display
    ev0 = f"EV {EVS[0]:+g}" if EVS and EVS[0] is not None else "EV ?"
    print(f"DARKEST bracket used at fusion time (#0, {ev0}):")
    display(_Img("darkest.jpg"))
    print("FUSED (Mertens, aligned brackets):")
    display(_Img("fused.jpg"))
except Exception:
    pass


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
    """SAM 2.1: union of pixel-precise masks for the given boxes.
    retina_masks=True -> masks come back at the image's native resolution (no blocky
    quarter-res upsampling), so the boundaries are as fine as SAM can produce."""
    if len(boxes) == 0:
        return np.zeros(bgr.shape[:2], bool)
    try:
        r = sam(bgr, bboxes=boxes.tolist(), retina_masks=True, verbose=False)
    except TypeError:
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
# ONLY window-related things are segmented: curtains/blinds, the window, its frame and
# glass doors — nothing else in the room is selected.
STRUCT = ["curtain", "blind", "window", "window frame", "glass door"]
# NO outdoor "view" concepts (sky/water/tree/umbrella) — Grounding DINO detecting "sky"
# INSIDE a room is unreliable and painted ceiling patches into the mask (the fake-sky /
# dark-ceiling-blob bug). The glass area comes from the window masks alone.
VIEW = []
# SYNONYM ensemble: each concept is detected with several phrasings and the boxes are
# UNIONED — one phrasing often misses an instance that another catches (recall boost).
SYN = {"window": ["window", "windowpane", "sliding glass window"],
       "curtain": ["curtain", "drape"],
       "blind": ["blind", "venetian blind", "window shade"]}
COLOURS = {"curtain": (200, 0, 200), "blind": (200, 0, 120), "window": (200, 200, 0),
           "window frame": (0, 140, 255), "glass door": (0, 90, 255),
           "sky": (255, 150, 0), "water": (255, 255, 0),
           "tree": (0, 200, 0), "umbrella": (0, 200, 200)}

# ---------------- VIEW IMAGE: uploaded BY YOU (no auto-select) ----------------
# Upload the ONE image whose window view you want composited into the glass — usually
# the darker bracket where the outdoor looks clearest to YOUR eye. It is resized and
# ECC-aligned to the same reference the brackets were aligned to, so it lines up exactly.
print(">>> Upload the DARKER / VIEW image now (the one whose window view you want):")
vup = files.upload()
vpath = list(vup.keys())[0]
viewimg = cv2.imread(vpath)
vh, vw = viewimg.shape[:2]
if vw != W:
    viewimg = cv2.resize(viewimg, (W, int(vh * W / vw)), interpolation=cv2.INTER_AREA)

# align the uploaded image to the brackets' alignment reference (the middle bracket)
_refimg = res[len(res) // 2]
try:
    s = min(1.0, 1000 / W)
    g_ref = cv2.equalizeHist(cv2.resize(cv2.cvtColor(_refimg, cv2.COLOR_BGR2GRAY), None,
                                        fx=s, fy=s, interpolation=cv2.INTER_AREA))
    g_v = cv2.equalizeHist(cv2.resize(cv2.cvtColor(viewimg, cv2.COLOR_BGR2GRAY), None,
                                      fx=s, fy=s, interpolation=cv2.INTER_AREA))
    warp = np.eye(3, dtype=np.float32)
    cv2.findTransformECC(g_ref, g_v, warp, cv2.MOTION_HOMOGRAPHY,
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
    S = np.diag([s, s, 1.0])
    Hm = (np.linalg.inv(S) @ warp.astype(np.float64) @ S).astype(np.float32)
    viewimg = cv2.warpPerspective(viewimg, Hm, (_refimg.shape[1], _refimg.shape[0]),
                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
    print(" * view image aligned to the brackets")
except cv2.error:
    print(" * WARNING: alignment failed — using the uploaded image as-is")

_t = _exptime(vpath)
_ev = round(math.log2(_t / med_t), 1) if (_t and med_t) else None
VIEW_DESC = f"your uploaded image '{vpath}'" + (f" (EV {_ev:+g})" if _ev is not None else "")
print(f"view source = {VIEW_DESC}  (mean {viewimg.mean():.0f})")
cv2.imwrite("view_bracket.jpg", viewimg, [cv2.IMWRITE_JPEG_QUALITY, 90])
try:
    from IPython.display import Image as _Img, display
    print("VIEW image (this exact photo will fill the glass):")
    display(_Img("view_bracket.jpg"))
except Exception:
    pass

os.makedirs("out", exist_ok=True)
concept_masks = {}
concept_boxes = {}          # the raw DINO rectangles — CELL 8 uses the window/door boxes
print("=== per-concept detection ===")
for name in STRUCT + VIEW:
    src = viewimg if name in VIEW else fused
    # synonym-ensemble detection: union the boxes from every phrasing of this concept
    all_boxes, top = [], 0.0
    for phrase in SYN.get(name, [name]):
        bxs, scs = detect(src, phrase)
        if len(bxs):
            all_boxes.append(bxs)
            top = max(top, float(scs.max()))
    boxes = np.vstack(all_boxes) if all_boxes else np.zeros((0, 4))
    m = segment(src, boxes)
    # NOTE: no brightness-based stripping here. The old `& ~SEETHRU` gate deleted the
    # SUNLIT parts of sheer curtains from the curtain mask, so they weren't subtracted
    # from the glass later and got composited DARK (the black-curtain bug). Curtain vs
    # umbrella is handled geometrically in CELL 8 (rod-hung components only).
    concept_masks[name] = m
    concept_boxes[name] = boxes
    ov = overlay(fused, m, COLOURS.get(name, (255, 255, 255)))   # always show on fused
    cv2.imwrite(f"out/overlay_{name.replace(' ', '_')}.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(f"out/mask_{name.replace(' ', '_')}.png", (m.astype(np.uint8) * 255))
    sc = f"{top:.2f}" if len(boxes) else "-"
    print(f"  {name:14s} on {'view' if name in VIEW else 'fused':8s}: "
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


# ============================== CELL 8 — enhance INDOOR only; CRISP outdoor view ======
# Enhance the indoor (room, curtains, frame). The GLASS / OUTDOOR VIEW gets NO HDR —
# instead it is composited from the SINGLE EV bracket where the view is CLEAREST
# (picked in CELL 5), so it is crisp and never washed out by the Mertens blend.
# Also shows the fused image so you can see the base going in.
H, Wd = fused.shape[:2]
k = max(3, int(Wd*0.004) | 1)

def _u(*names):
    m = np.zeros((H, Wd), np.float32)
    for n in names:
        cm = concept_masks.get(n)
        if cm is not None:
            m = np.clip(m + (cm > 0).astype(np.float32), 0, 1)
    return m

# =================== GLASS-ONLY MASK — pure geometry, no brightness gates ===============
# Root-cause redesign. Every previous failure came from (a) detecting outdoor concepts
# (sky/tree) INDOORS -> ceiling blobs, and (b) brightness thresholds -> holes on shaded
# view content + sunlit curtains stripped. Now: only the window/glass-door SAM masks
# define WHERE glass can be; curtains and frame are removed by GEOMETRY.

# 1) WINDOW area = SAM masks for window + glass door ONLY. Erode a small lip so the
#    OUTER frame edge is excluded from the start.
WIN = _u("window", "glass door") > 0.5
lip = max(3, int(Wd * 0.004) | 1)
WIN_IN = cv2.erode(WIN.astype(np.uint8), np.ones((lip, lip), np.uint8)).astype(bool)

# 2) CURTAINS: keep only ROD-HUNG components (top of the component in the upper part of
#    the image). Real curtains hang from a rod; a pool umbrella / outdoor chair that DINO
#    mislabelled 'curtain' is a mid-window blob and is ignored. NO brightness test, so a
#    sunlit sheer curtain stays a curtain and can never be composited dark again.
curt_raw = (_u("curtain", "blind") > 0.5).astype(np.uint8)
ncc, lab, st, _ = cv2.connectedComponentsWithStats(curt_raw)
CURT = np.zeros(curt_raw.shape, bool)
for i in range(1, ncc):
    if st[i, cv2.CC_STAT_TOP] < 0.45 * H and st[i, cv2.CC_STAT_AREA] > 0.0005 * H * Wd:
        CURT |= (lab == i)

# 3) FRAME / MULLIONS: they are THIN, ACHROMATIC structures inside the window. A wide
#    morphological OPEN erases thin bars but keeps large blobs — so grey pavement, signs
#    or shaded patio (large achromatic blobs) STAY in the view (no holes), while every
#    white or dark bar is removed regardless of its brightness (white frames too).
v = viewimg.astype(np.float32) / 255; vl = v.mean(2); vc = v.max(2) - v.min(2)
achro = (vc < 0.07) & WIN_IN
bar_k = max(5, int(Wd * 0.014) | 1)
blobs = cv2.morphologyEx(achro.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((bar_k, bar_k), np.uint8)).astype(bool)
FRAME = (achro & ~blobs) | (achro & (vl < 0.15))     # thin bars (any brightness) + dark bars

# GLASS = window interior minus frame minus curtains — full coverage, no holes.
glass_view = WIN_IN & (~FRAME) & (~CURT)
glass_view = cv2.morphologyEx(glass_view.astype(np.float32), cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))
# EDGE-AWARE feather (guided filter): the boundary snaps to the real glass/frame/curtain
# edge in the fused image; clamped so it can never leave the window masks.
glass_view = _feather(glass_view, fused.astype(np.float32)/255)
glass_view = glass_view * WIN.astype(np.float32)
print(f"protected glass-only view = {(glass_view>0.5).mean()*100:.1f}%  (rest gets HDR)")

# SHOW the VERIFIED segments — all in one colour-coded map (these are CELL 8's cleaned
# masks, AFTER the see-through verification, not the raw CELL 5 detections):
#   CYAN = glass/outdoor view · ORANGE = frame/mullions · MAGENTA = curtains/blinds
seg_vis = fused.astype(np.float32)
for msk, col in ((glass_view > 0.5, (220, 220, 0)),
                 (FRAME, (0, 140, 255)),
                 (CURT, (200, 0, 200))):
    a = msk.astype(np.float32)[..., None] * 0.5
    tint = np.zeros_like(seg_vis); tint[:] = col
    seg_vis = seg_vis * (1 - a) + tint * a
cv2.imwrite("segments_verified.jpg", seg_vis.clip(0, 255).astype("uint8"),
            [cv2.IMWRITE_JPEG_QUALITY, 92])

# and the protect mask alone, for the final check:
prot_vis = overlay(fused, glass_view > 0.5, (220, 220, 0))
cv2.imwrite("protected_view.jpg", prot_vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
try:
    from IPython.display import Image as _Img, display
    print("VERIFIED segments — cyan glass · orange frame · magenta curtains:")
    display(_Img("segments_verified.jpg"))
    print("PROTECTED outdoor view (cyan) — HDR applies to everything else:")
    display(_Img("protected_view.jpg"))
except Exception:
    pass

# ---- enhance the whole (indoor) image; the view is excluded from the measurements ----
img = fused.astype(np.float32)/255
ex = (glass_view > 0.3).astype(np.float32)
boost = concept_masks.get("wall")
protect = concept_masks.get("floor")
img = _wb(img, exclude=ex)
img = _levels(img, exclude=ex)
img = _expose(img, 0.70)
img = _neutralize(img, boost=None if boost is None else boost.astype(np.float32),
                  protect=None if protect is None else protect.astype(np.float32))
img = _match_white(img)
img = _scurve(img, 0.05)
img = _sharpen(img)

# ---- OUTDOOR VIEW = the image YOU uploaded in CELL 5 ----
# NOT the Mertens blend (blown brighter brackets wash it out) and NOT enhanced — your
# chosen image supplies the view content, gently lifted so it sits naturally next to
# the enhanced room.
print(f"outdoor view composited from {VIEW_DESC}")
try:
    from IPython.display import Image as _Img, display
    print("THIS uploaded image is being blended into the glass:")
    display(_Img("view_bracket.jpg"))
except Exception:
    pass
vsrc = viewimg.astype(np.float32)/255
vlum = vsrc.mean(2)
sel = glass_view > 0.5
med = float(np.median(vlum[sel])) if sel.any() else 0.0
if 1e-3 < med < 0.58:                       # gentle lift only if the view sits dark
    vsrc = np.clip(vsrc, 0, 1) ** float(np.clip(np.log(0.58)/np.log(med), 0.6, 1.0))
gv = glass_view[..., None]
img = img*(1-gv) + vsrc*gv

result = (img*255).clip(0, 255).astype("uint8")
cv2.imwrite("hdr_result.jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
try:
    from IPython.display import Image as _Img, display
    print("FUSED (base going in):"); display(_Img("fused.jpg"))
    print(f"FINAL — indoor enhanced + crisp outdoor view from {VIEW_DESC}:")
    display(_Img("hdr_result.jpg"))
except Exception:
    pass
files.download("fused.jpg")
files.download("segments_verified.jpg")
files.download("protected_view.jpg")
files.download("hdr_result.jpg")
print("done — indoor (curtains + frame + room) enhanced; outdoor view from your image")
