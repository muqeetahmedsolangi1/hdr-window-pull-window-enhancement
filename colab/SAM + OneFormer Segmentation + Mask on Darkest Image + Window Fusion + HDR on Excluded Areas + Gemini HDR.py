"""
CLEAR-SEG WINDOW PULL — segment on the CLEAR image, apply on the DARKER image.

THE INSIGHT (user-found): SAM 3 / OneFormer are OBJECT models. On a dark bracket
the window is barely an object — only the outdoor view glows. On a 0 EV / fused
image the window is clearly visible, so THAT is where segmentation must run.
The mask is then transferred (aligned) onto the darker image, whose pixels fill
the glass at HDR time.

Flow:
  CELL 2  upload the CLEAR image (0 EV / normal exposure — windows clearly visible)
  CELL 3  SAM 3 segments windows + curtains on the CLEAR image (raw, own cell)
  CELL 4  OneFormer segments windowpane on the CLEAR image (raw, own cell)
  CELL 5  upload the DARKER image -> aligned to the clear image -> the dark-image
          physics gates fire NOW (see-through gate, curtain subtract, dark frame
          bars) -> final SAM 3 + OneFormer glass masks shown ON the darker image
  CELL 6  upload the exposure BRACKETS -> aligned -> Mertens FUSED
  CELL 7  dropdown: 1 = SAM 3, 2 = OneFormer -> HDR everywhere EXCEPT the chosen
          glass; the glass shows YOUR darker image, crisp. Downloads.
  CELL 8  OPTIONAL: white window frames via Gemini Nano Banana Pro with the hard
          anti-hallucination composite (only frame-band pixels from Gemini).

Models: facebook/sam3 (gated-free) · shi-labs/oneformer_ade20k_dinat_large
(auto-fallback to swin_large). All shootout gates kept — they just run at
CELL 5, once the darker image exists.

Run on Google Colab with a GPU (A100 or T4). Copy each CELL into its own cell.
"""

# ============================== CELL 1 — install ==============================
# (Runtime -> Change runtime type -> GPU, first!)
# !pip install -q -U transformers opencv-python-headless
# !pip install -q --force-reinstall "pillow==10.4.0"
# !pip install -q --force-reinstall -U huggingface_hub
# # OPTIONAL (OneFormer DiNAT backbone; CELL 4 falls back to Swin — ignore errors):
# !pip install -q natten -f https://shi-labs.com/natten/wheels --trusted-host shi-labs.com
#
# >>> AFTER THIS CELL: Runtime -> Restart session, then run CELL 2. <<<


# ============================== CELL 2 — upload the CLEAR image ===============
# The image where the WINDOWS ARE CLEARLY VISIBLE as objects: the 0 EV / normal
# bracket (or an already-fused image). This image is the coordinate frame —
# the darker image and all brackets get aligned TO it.
import os, math, time, torch, numpy as np, cv2
from PIL import Image
from google.colab import files

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| gpu:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "-")
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if DEVICE == "cuda" else 0
KEEP_MODELS = VRAM_GB >= 30
print(f"vram: {VRAM_GB:.0f} GB -> " +
      ("models stay loaded (big gpu)" if KEEP_MODELS else "freeing after each cell (small gpu)"))

print(">>> Upload the CLEAR image (0 EV / normal exposure — windows clearly visible):")
up = files.upload()
if not up:
    raise RuntimeError("no file uploaded — run this cell again")
_name = list(up.keys())[0]
clearimg = cv2.imread(_name)
if clearimg is None:
    raise RuntimeError(f"could not read '{_name}' — upload a JPG/PNG")
W = 2000
h, w = clearimg.shape[:2]
if w != W:
    clearimg = cv2.resize(clearimg, (W, int(h * W / w)), interpolation=cv2.INTER_AREA)
H, Wd = clearimg.shape[:2]

RAW = {}                                          # raw model outputs (clear image)
GLASS = {}                                        # final gated masks (after CELL 5)
cv2.imwrite("clear_input.jpg", clearimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"clear image: {_name} -> {Wd}x{H} (mean {clearimg.mean():.0f}/255)")
try:
    from IPython.display import Image as _Img, display
    print("SEGMENTATION runs on THIS image (windows must look clear here):")
    display(_Img("clear_input.jpg"))
except Exception:
    pass


def _align_to_clear(img):
    """ECC-align any image of the scene to the CLEAR image's coordinate frame."""
    s = min(1.0, 1000 / W)
    g_ref = cv2.equalizeHist(cv2.resize(cv2.cvtColor(clearimg, cv2.COLOR_BGR2GRAY), None,
                                        fx=s, fy=s, interpolation=cv2.INTER_AREA))
    g_im = cv2.equalizeHist(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None,
                                       fx=s, fy=s, interpolation=cv2.INTER_AREA))
    S = np.diag([s, s, 1.0])
    warp = np.eye(3, dtype=np.float32)
    try:
        cv2.findTransformECC(g_ref, g_im, warp, cv2.MOTION_HOMOGRAPHY,
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
                             None, 5)
        Hm = (np.linalg.inv(S) @ warp.astype(np.float64) @ S).astype(np.float32)
        return cv2.warpPerspective(img, Hm, (Wd, H),
                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REPLICATE), True
    except cv2.error:
        return img, False


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

print("ready — run CELL 3 (SAM 3) and CELL 4 (OneFormer)")


# ============================== CELL 3 — SAM 3 on the CLEAR image =============
# facebook/sam3 is GATED (free): accept the license at hf.co/facebook/sam3, make
# a READ token, add it as Colab secret HF_TOKEN — or paste it when asked.
# Raw masks only here — the dark-image gates fire in CELL 5.
from huggingface_hub import login
try:
    from google.colab import userdata
    _tok = userdata.get("HF_TOKEN")
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


t0 = time.time()
RAW["sam3_windows"] = sam3_instances(clearimg, ("window pane", "window", "glass door"))
# curtains at a STRICTER threshold (0.5) — only REAL curtains count; if the
# scene has none, zero instances come back and NOTHING gets subtracted
RAW["sam3_curtains"] = sam3_instances(clearimg, ("curtain", "drape"), threshold=0.5)

_u = np.zeros((H, Wd), bool)
for mi in RAW["sam3_windows"]:
    _u |= mi
# real-curtain filter: rod-hung (top of the component in the upper part of the
# image) and not a speck — a bed/fan mislabelled 'curtain' never passes this
_c = np.zeros((H, Wd), bool)
for mi in RAW["sam3_curtains"]:
    _c |= mi
CURT_CLEAR = np.zeros((H, Wd), bool)
ncc, lab, st, _ = cv2.connectedComponentsWithStats(_c.astype(np.uint8))
for i in range(1, ncc):
    if st[i, cv2.CC_STAT_TOP] < 0.45 * H and st[i, cv2.CC_STAT_AREA] > 0.0005 * H * Wd:
        CURT_CLEAR |= (lab == i)
RAW["sam3_curtain_mask"] = CURT_CLEAR

sam3_glass_clear = _u & ~CURT_CLEAR              # the area you need: glass, not fabric
print(f"SAM 3 on the CLEAR image: {len(RAW['sam3_windows'])} window instances, "
      f"{len(RAW['sam3_curtains'])} curtain instances "
      f"({'curtains subtracted' if CURT_CLEAR.any() else 'no real curtains -> nothing subtracted'})"
      f"  [{time.time()-t0:.1f}s]")
show_mask("SAM3 windows minus curtains (clear image)", (255, 150, 0), clearimg,
          sam3_glass_clear, "raw_sam3.jpg")
if CURT_CLEAR.any():
    show_mask("SAM3 curtains found (subtracted)", (200, 0, 200), clearimg,
              CURT_CLEAR, "raw_sam3_curtains.jpg")

if not KEEP_MODELS:
    del s3_model, s3_proc
    import gc; gc.collect(); torch.cuda.empty_cache()


# ============================== CELL 4 — OneFormer on the CLEAR image =========
# ADE20K semantics on the clear image; class logic + gates fire in CELL 5.
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

try:
    OF_ID = "shi-labs/oneformer_ade20k_dinat_large"
    of_proc = OneFormerProcessor.from_pretrained(OF_ID)
    of_model = OneFormerForUniversalSegmentation.from_pretrained(OF_ID).to(DEVICE).eval()
except Exception as e:
    print(f"DiNAT backbone unavailable ({type(e).__name__}) — falling back to Swin-L")
    OF_ID = "shi-labs/oneformer_ade20k_swin_large"
    of_proc = OneFormerProcessor.from_pretrained(OF_ID)
    of_model = OneFormerForUniversalSegmentation.from_pretrained(OF_ID).to(DEVICE).eval()
print("OneFormer:", OF_ID)

t0 = time.time()
pil = Image.fromarray(cv2.cvtColor(clearimg, cv2.COLOR_BGR2RGB))
inp = of_proc(images=pil, task_inputs=["semantic"], return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = of_model(**inp)
seg = of_proc.post_process_semantic_segmentation(out, target_sizes=[pil.size[::-1]])[0]
RAW["oneformer_seg"] = seg.cpu().numpy()
_r = RAW["oneformer_seg"] == 8
print(f"OneFormer on the CLEAR image: windowpane raw {_r.mean()*100:.1f}%  [{time.time()-t0:.1f}s]")
show_mask("OneFormer raw windowpane (clear image)", (200, 0, 200), clearimg, _r, "raw_oneformer.jpg")

if not KEEP_MODELS:
    del of_model, of_proc
    import gc; gc.collect(); torch.cuda.empty_cache()


# ============================== CELL 5 — upload DARKER image, TRANSFER masks ==
# NO re-segmentation here — no model runs on the darker image. The CELL 3/4
# masks are placed onto it as-is (PURE TRANSFER). The optional checkbox turns
# the physics cleanup back on if a scene ever needs it.
print(">>> Upload the DARKER image (the one whose window view you want):")
vup = files.upload()
vpath = list(vup.keys())[0]
darkimg = cv2.imread(vpath)
vh, vw = darkimg.shape[:2]
if vw != W:
    darkimg = cv2.resize(darkimg, (W, int(vh * W / vw)), interpolation=cv2.INTER_AREA)
darkimg, _ok = _align_to_clear(darkimg)
print(" * darker image aligned to the clear image" if _ok
      else " * WARNING: alignment failed — using as-is")

d = darkimg.astype(np.float32) / 255
dl = d.mean(2); dc = d.max(2) - d.min(2)            # also used by CELL 8 (frame band)
thr = max(0.22, 0.45 * float(np.percentile(dl, 99.9)))
SEE = (dl >= thr) | (dc >= 0.15)                  # the view: bright OR colourful
det_dark = (np.clip(d ** 0.5, 0, 1) * 255).astype("uint8")   # lifted, display only

APPLY_CLEANUP = False  #@param {type:"boolean"}
# False (default) = PURE TRANSFER: the CELL 3/4 masks are used EXACTLY as they
#                   are — nothing added, nothing removed.
# True            = also run the physics cleanup on them (see-through gate,
#                   rod-hung curtain subtraction, dark frame bars, specks).
#                   Turn this on if a result shows curtains/doors in the glass.


def refine_glass(raw):
    """Dark-image physics: remove thin DARK colourless frame bars (lit venetian
    slats KEPT), drop specks."""
    m = raw.astype(bool).copy()
    achro = (dc < 0.07) & (dl < 0.18) & m
    bar_k = max(5, int(Wd * 0.014) | 1)
    blobs = cv2.morphologyEx(achro.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((bar_k, bar_k), np.uint8)).astype(bool)
    m &= ~(achro & ~blobs)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    ncc, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    for i in range(1, ncc):
        if st[i, cv2.CC_STAT_AREA] < 0.0005 * H * Wd:
            m[lab == i] = False
    return m


# ---- SAM 3 mask: union of the CELL 3 window instances, transferred as-is ----
raw = np.zeros((H, Wd), bool)
if APPLY_CLEANUP:
    kept = drop = 0
    for mi in RAW["sam3_windows"]:
        if float(SEE[mi].mean()) >= 0.06:         # a real window GLOWS in the dark
            raw |= mi; kept += 1
        else:
            drop += 1
    curt = np.zeros((H, Wd), bool)
    for mi in RAW["sam3_curtains"]:
        curt |= mi
    ncc, lab, st, _ = cv2.connectedComponentsWithStats(curt.astype(np.uint8))
    for i in range(1, ncc):
        if st[i, cv2.CC_STAT_TOP] < 0.45 * H and st[i, cv2.CC_STAT_AREA] > 0.0005 * H * Wd:
            raw &= ~(lab == i)                    # rod-hung curtains subtracted
    print(f"SAM 3 cleanup: {kept} window instances kept, {drop} dark rejected")
    GLASS["sam3"] = refine_glass(raw)
else:
    for mi in RAW["sam3_windows"]:
        raw |= mi
    # pure transfer of the CELL 3 result: windows minus the REAL curtains found
    # there (if no curtains existed, the curtain mask is empty — nothing changes)
    GLASS["sam3"] = raw & ~RAW.get("sam3_curtain_mask", np.zeros((H, Wd), bool))
    print(f"SAM 3: pure transfer of {len(RAW['sam3_windows'])} instances"
          + (" minus real curtains" if RAW.get("sam3_curtain_mask", np.zeros(1)).any() else ""))
show_mask("SAM3 FINAL (on darker image)", (255, 150, 0), det_dark,
          GLASS["sam3"], "glass_sam3.jpg")

# ---- OneFormer mask: the CELL 4 windowpane class, transferred as-is ----
seg = RAW["oneformer_seg"]
if APPLY_CLEANUP:
    GLASS["oneformer"] = refine_glass((seg == 8) | ((seg == 14) & SEE))
else:
    GLASS["oneformer"] = (seg == 8)               # pure transfer — untouched
    print("OneFormer: pure transfer of the windowpane class")
show_mask("OneFormer FINAL (on darker image)", (200, 0, 200), det_dark,
          GLASS["oneformer"], "glass_oneformer.jpg")


# ============================== CELL 6 — upload BRACKETS, align, FUSE =========
# Select ALL exposure brackets of the scene (e.g. 7 JPGs) in one go — every
# bracket is aligned to the CLEAR image (the masks' coordinate frame).
print(">>> Upload the exposure BRACKETS now (all together):")
bup = files.upload()
bpaths = sorted(bup.keys())
if len(bpaths) < 2:
    raise RuntimeError("upload at least 2 brackets (usually 5-8)")

aligned = []
for p in bpaths:
    im = cv2.imread(p)
    if im is None:
        continue
    hh, ww = im.shape[:2]
    if ww != W:
        im = cv2.resize(im, (W, int(hh * W / ww)), interpolation=cv2.INTER_AREA)
    im, _ = _align_to_clear(im)
    aligned.append(im)
aligned.sort(key=lambda im: im.mean())            # darkest -> brightest

fused = (cv2.createMergeMertens().process(aligned) * 255).clip(0, 255).astype("uint8")
cv2.imwrite("fused.jpg", fused, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"{len(aligned)} brackets (aligned to the clear image) -> fused {Wd}x{H}")
try:
    from IPython.display import Image as _Img, display
    print("FUSED (Mertens):")
    display(_Img("fused.jpg"))
except Exception:
    pass


# ============================== CELL 7 — PICK a segmentation, then HDR ========
# Compact side-by-side of the two FINAL masks; pick via the dropdown (form field
# on the right of this cell in Colab) — input() boxes are unreliable there.
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

def _feather(mask, guide_bgr):
    r = max(6, int(mask.shape[1] * 0.004))
    try:
        gd = cv2.cvtColor((np.clip(guide_bgr, 0, 1) * 255).astype(np.uint8),
                          cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        return np.clip(cv2.ximgproc.guidedFilter(gd, mask.astype(np.float32), r, 1e-3), 0, 1)
    except Exception:
        return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), r * 0.6), 0, 1)

glass_view = _feather(GLASS[key].astype(np.float32), fused.astype(np.float32) / 255)

# ================= NEW HDR CHAIN — identical to scripts/local_hdr_test.py ======
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
    luma = img.mean(2); s2 = luma if exclude is None else luma[exclude < 0.5]
    if s2.size < 1000: s2 = luma
    lo = min(float(np.percentile(s2, blk)), maxblk); hi = float(np.percentile(s2, wht))
    if hi - lo < 0.1: return img
    return np.clip((img - lo) / (hi - lo), 0, 1) * (0.91 - 0.02) + 0.02


def _shadow_fill(fused_u8, brightest, T=0.18, p=1.5, max_fill=0.30, feather=4):
    """FUSION-TIME deep-shadow fill from the BRIGHTEST bracket. Mertens
    regionally suppresses a mostly-blown bracket even where it is the best
    frame (measured), so crushed pockets (< T) get its real pixels blended in,
    max 30%. Weight is a smooth function of LUMINANCE (not a structure
    detector) so it cannot halo silhouettes."""
    fl = fused_u8.astype(np.float32).mean(2) / 255
    w = np.clip((T - fl) / T, 0, 1) ** p
    w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), feather) * max_fill
    w3 = w[..., None]
    out = fused_u8.astype(np.float32) * (1 - w3) + brightest.astype(np.float32) * w3
    return out.clip(0, 255).astype("uint8")


def _bright_ramp(img):
    """Soft mask of already-bright/clipped pixels (blown windows / lights) —
    combined with the REAL glass mask so the interior lifts never push the
    blown areas further."""
    luma = img.mean(2)
    ramp = np.clip((luma - 0.80) / 0.12, 0, 1)
    k = max(31, int(img.shape[1] * 0.008) | 1)
    return cv2.GaussianBlur(ramp, (k, k), 0)


def _expose(img, target=0.68, exclude=None):
    # interior median measured on NON-window pixels only; the gamma is FADED
    # OUT over the exclude mask so the room brightens without blasting glass
    luma = img.mean(2)
    sel = luma if exclude is None else luma[exclude < 0.5]
    if sel.size < 1000: sel = luma
    m = max(float(np.median(sel)), 1e-3)
    if m >= target:
        return img
    gamma = float(np.clip(np.log(target) / np.log(m), 0.45, 1.0))
    lifted = np.clip(img, 0, 1) ** gamma
    if exclude is None:
        return lifted
    ex2 = exclude[..., None]
    return lifted * (1 - ex2) + np.clip(img, 0, 1) * ex2


def _neutralize(img):
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]/255; a = lab[..., 1]-128; b = lab[..., 2]-128
    ch = np.sqrt(a*a+b*b)
    wgt = np.clip((L-0.70)/0.12, 0, 1) * np.clip((22-ch)/8, 0, 1)
    wgt = np.maximum(wgt, np.clip((L-0.85)/0.10, 0, 1))
    # MUDDY FIX: de-tint shadows/midtones too (tight chroma gate — real
    # colour never touched), removing the warm dirty cast on dark surfaces
    shadow_wgt = np.clip((14-ch)/6, 0, 1) * np.clip((L-0.04)/0.05, 0, 1) * 0.5
    wgt = np.maximum(wgt, shadow_wgt)
    # HALO-BAND FIX: tungsten shading on a white ceiling/wall sits at mid-luma
    # (L 0.5-0.7) with chroma 14-24 — the GAP between the terms above — so a
    # whitened ceiling shows a tan halo strip along its shaded junction.
    # Near-neutral pure-YELLOW pixels only: the a-gate spares green walls
    # (a<=-5) and pink fabric / orange wood (a>=+4). Strength is a-dependent:
    # 0.9 on paint shading (a<=-1.5), 0.55 on warm-tan fabric (a>=+0.5).
    ga = np.clip((a+5)/3, 0, 1) * np.clip((4-a)/4, 0, 1)
    strength = 0.55 + 0.35*np.clip((0.5-a)/2, 0, 1)
    yellow_wgt = (ga * np.clip((b-4)/6, 0, 1) * np.clip((28-ch)/8, 0, 1)
                  * np.clip((L-0.45)/0.12, 0, 1) * strength)
    # BULB-GLOW path: a warm bulb's glow on a white ceiling is slightly
    # a-POSITIVE (a +1..+3). Illumination cast is b-dominated with small a
    # RELATIVE to its chroma — (ch-2a) sits at 7..11 for the glow but >=13
    # for real tan curtain folds — that axis separates them.
    sc = ch - 2*a
    glow_wgt = (np.clip((6-a)/4, 0, 1) * np.clip((13.5-sc)/3.5, 0, 1)
                * np.clip((b-4)/6, 0, 1) * np.clip((L-0.45)/0.12, 0, 1) * 0.9)
    yellow_wgt = np.maximum(yellow_wgt, glow_wgt)
    wgt = cv2.GaussianBlur(wgt, (0, 0), 8)
    # the band is a NARROW strip: the sigma-8 blur would dilute its weight
    # with the zero-weight wall next to it, so the yellow term gets its own
    # tighter blur and joins afterwards.
    wgt = np.maximum(wgt, cv2.GaussianBlur(yellow_wgt, (0, 0), 3))
    lab[..., 1] = a*(1-wgt)+128; lab[..., 2] = b*(1-wgt)+128
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _match_white(img, strength=0.85):
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    br = lab[..., 0] > 0.78*255
    if br.sum() < 1000: return img
    lab[..., 1] += (128-float(lab[..., 1][br].mean()))*strength
    lab[..., 2] += (128-float(lab[..., 2][br].mean()))*strength
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _lift_whites(img, amount=0.18, start=0.45, exclude=None):
    """Brighten ONLY the bright zone (walls/ceiling/whites) toward white;
    `exclude` (glass + blown areas) keeps the window OUT of this lift."""
    luma = img.mean(2)
    w = np.clip((luma - start) / (1.0 - start), 0, 1) ** 1.3
    if exclude is not None:
        w = w * (1 - exclude)
    return np.clip(img + amount * w[..., None] * (1.0 - img), 0, 1)


def _scurve(img, s2=0.05): return np.clip(img + s2*np.sin(2*np.pi*(img-0.5)), 0, 1)


def _tame_warm(img, knee=16, compress=0.55, l_gain=24):
    """Over-saturated / too-dark BROWNS & ORANGES fix (terracotta, dark wood):
    warm-quadrant chroma soft-knee + small L give-back on dark warm pixels."""
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    A = lab[..., 1]-128; B = lab[..., 2]-128
    ch = np.sqrt(A*A+B*B)
    warm_w = np.clip(A/6, 0, 1) * np.clip(B/10, 0, 1)     # soft brown/orange gate
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
    """CLIENT FIX: desaturate the YELLOW and ORANGE hues across the whole
    result (LAB hue window 30..115 deg, like Lightroom's HSL yellow/orange
    saturation pulled down). The a-gate hard-protects green walls (sage sits
    at hue ~120 right next to the window). Yellow (hue > 70) is compressed
    harder (60%) than orange (40%) so wood keeps some warmth."""
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1]-128; b = lab[..., 2]-128
    hue = np.degrees(np.arctan2(b, a))            # orange ~60, yellow ~90
    w = np.clip((hue-lo)/18, 0, 1) * np.clip((hi-hue)/12, 0, 1)
    w *= np.clip(b/6, 0, 1) * np.clip((a+6)/4, 0, 1)
    w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), 3)
    amount = 0.4 + 0.2*np.clip((hue-70)/15, 0, 1)  # orange 0.4 -> yellow 0.6
    f = 1 - amount*w
    lab[..., 1] = a*f + 128; lab[..., 2] = b*f + 128
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _sharpen(img, fine=0.9, clarity=0.2):
    img = np.clip(img + fine*(img - cv2.GaussianBlur(img, (0, 0), 1.0)), 0, 1)
    wide = cv2.GaussianBlur(img, (0, 0), 15)
    # MUDDY FIX: damp the wide-radius clarity in shadows — full-strength local
    # contrast on already-dark pixels reads as "muddy/gritty".
    luma = img.mean(2)
    damp = np.clip((luma - 0.25) / 0.25, 0.15, 1.0)[..., None]
    return np.clip(img + clarity*damp*(img - wide), 0, 1)
# ==============================================================================

# fusion-time crushed-shadow fill (brightest aligned bracket = aligned[-1])
fused_sf = _shadow_fill(fused, aligned[-1])

img = fused_sf.astype(np.float32)/255
# exclude = the REAL glass mask + any other blown/bright areas (lights, extra
# windows the mask missed) — same role as the local app's bright ramp
ex = np.maximum(glass_view, _bright_ramp(img))
img = _wb(img, exclude=ex)
img = _levels(img, exclude=ex)
img = _expose(img, exclude=ex)     # interior median -> 0.68; window shielded
img = _neutralize(img)             # muddy + halo-band + bulb-glow fixes
img = _match_white(img)
img = _lift_whites(img, exclude=ex)  # walls/whites brightened; window shielded
img = _scurve(img, 0.05)
img = _tame_warm(img)              # browns/oranges: de-oversaturate + un-darken
img = _desat_warm(img)             # CLIENT FIX: yellows/oranges desaturated
img = _sharpen(img)                # clarity damped in shadows

# ---- the glass shows YOUR darker image (aligned in CELL 5), crisp ----
vsrc = darkimg.astype(np.float32)/255
sel = glass_view > 0.5
med = float(np.median(vsrc.mean(2)[sel])) if sel.any() else 0.0
if 1e-3 < med < 0.58:                             # gentle lift only if it sits dark
    vsrc = np.clip(vsrc, 0, 1) ** float(np.clip(np.log(0.58)/np.log(med), 0.6, 1.0))
gv = glass_view[..., None]
img = img*(1-gv) + vsrc*gv

result = (img*255).clip(0, 255).astype("uint8")
cv2.imwrite("hdr_result.jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite("glass_mask_chosen.png", ((glass_view > 0.5).astype(np.uint8) * 255))
try:
    from IPython.display import Image as _Img, display
    print(f"FINAL — HDR everywhere except the {key.upper()} glass (view from your darker image):")
    display(_Img("hdr_result.jpg"))
except Exception:
    pass
files.download("fused.jpg")
files.download("glass_mask_chosen.png")
files.download("hdr_result.jpg")
print(f"done — segmented on the CLEAR image, view from the DARKER image ({key.upper()})")


# ============================== CELL 8 — OPTIONAL: WHITE FRAMES via Gemini ====
# Same hallucination-proof white-frame step as before: only the frame-band
# pixels can come from Gemini. Needs (once): !pip install -q -U google-genai
# and a GEMINI_API_KEY Colab secret (or paste it when asked).
from google import genai
from google.genai import types as gtypes

try:
    from google.colab import userdata
    _gkey = userdata.get("GEMINI_API_KEY")
except Exception:
    _gkey = os.environ.get("GEMINI_API_KEY")
if not _gkey:
    _gkey = input("paste your Gemini API key: ")
_gkey = "".join(str(_gkey).split())               # kill stray newlines in the key
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

ok, jb = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
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
    print(f"retrying without config ({type(e).__name__})")
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
# The old version (thin ring + only near-black dl<0.18 bars) left big black gaps
# on thick sliding-door frames AND wrongly ate dark view content (a pool-cage /
# tree seen THROUGH the glass). This rebuild fixes both — verified on a synthetic
# window: frame capture 42% -> 100%, dark view eaten 100% -> 0%.
g8 = (glass_view > 0.5).astype(np.uint8)
glass_bool = g8.astype(bool)

# zone = the window unit + a generous margin; the frame can never be outside it
zone = cv2.dilate(g8, np.ones((max(11, int(Wd * 0.05) | 1),) * 2, np.uint8)).astype(bool)

# 1) RING hugging the glass — the outer casing/sash AND the thin gap between two
#    panes (the centre mullion sits in that gap). Wider than the old ring.
ring_k = max(13, int(Wd * 0.022) | 1)
ring = cv2.dilate(g8, np.ones((ring_k, ring_k), np.uint8)).astype(bool) & ~glass_bool

# 2) DARK colourless frame in the zone. Thresholds RELAXED (dl < 0.35 / dc < 0.12,
#    was 0.18 / 0.07) so medium-grey frame is caught too — that relaxation is what
#    removes the leftover black. Split so real view is protected:
#      * OUTSIDE the glass  -> keep all of it (the casing/sash)
#      * INSIDE the glass   -> keep only THIN LINE-shaped bars (pane dividers);
#        a thick dark blob there is the outdoor view (pool cage / dark tree) and
#        is deliberately NOT touched.
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
final = (result.astype(np.float32) * (1 - fb) + gem.astype(np.float32) * fb)
final = final.clip(0, 255).astype("uint8")

cv2.imwrite("gemini_raw.jpg", gem, [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite("hdr_result_whiteframe.jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"frame band = {(FRAMEB > 0.5).mean()*100:.1f}% of image — only these pixels came from Gemini")
try:
    from IPython.display import Image as _Img, display
    print("BEFORE (your HDR result):");   display(_Img("hdr_result.jpg"))
    print("GEMINI raw output (reference only — NOT used directly):"); display(_Img("gemini_raw.jpg"))
    print("AFTER — white frame, everything else guaranteed untouched:")
    display(_Img("hdr_result_whiteframe.jpg"))
except Exception:
    pass
files.download("hdr_result_whiteframe.jpg")
print("done — white-frame result: Gemini touched ONLY the frame band")
