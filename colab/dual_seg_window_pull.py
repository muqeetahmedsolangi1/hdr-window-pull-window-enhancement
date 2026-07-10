"""
DUAL-SEG WINDOW PULL — segment first (SAM 3 *and* OneFormer, separate cells),
then fuse the brackets, then YOU pick which segmentation drives the window pull.

Flow (exactly as requested):
  CELL 2  upload the DARKER image first
  CELL 3  SAM 3 segments the window glass  (its own cell, own overlay)
  CELL 4  OneFormer segments the window glass (its own cell, own overlay)
  CELL 5  upload the 7 exposure BRACKETS -> aligned to the darker image -> FUSED
  CELL 6  shows both masks side by side and ASKS: "1 = SAM 3, 2 = OneFormer?"
          -> the chosen glass area is LEFT ALONE (crisp view from your darker
          image), HDR finishing is applied to everything else. Downloads.
  CELL 7  OPTIONAL: sends the result to Gemini Nano Banana Pro to paint the
          window FRAME white — hallucination is blocked by a hard mask
          composite (only frame-band pixels can come from Gemini).

Models:
  SAM 3     facebook/sam3 (gated-free: accept license + HF token)   [SAM license]
  OneFormer shi-labs/oneformer_ade20k_dinat_large, auto-fallback    [research]
            to oneformer_ade20k_swin_large when natten is missing

Shootout lessons baked into BOTH pipelines:
  * models look at a gamma-LIFTED copy; all physics on the ORIGINAL dark image
  * SAM 3 instances must GLOW in the dark bracket (see gate) — doors/walls out
  * SAM 3 curtains ("curtain"/"drape", rod-hung only) are subtracted;
    OneFormer excludes curtains natively (they have their own ADE20K class)
  * frame bars = thin DARK colourless lines only — lit venetian slats are KEPT

Run on Google Colab with a GPU (A100 or T4). Copy each CELL into its own cell.
"""

# ============================== CELL 1 — install ==============================
# (Runtime -> Change runtime type -> GPU, first!)
# !pip install -q -U transformers opencv-python-headless
# !pip install -q --force-reinstall "pillow==10.4.0"
# !pip install -q --force-reinstall -U huggingface_hub
# # OPTIONAL (OneFormer's strongest DiNAT backbone; CELL 4 falls back to Swin
# # automatically — if this errors, IGNORE it):
# !pip install -q natten -f https://shi-labs.com/natten/wheels --trusted-host shi-labs.com
#
# >>> AFTER THIS CELL: Runtime -> Restart session, then run CELL 2. <<<


# ============================== CELL 2 — upload the DARKER image ==============
# The darker exposure where the window view is clear. Everything else (brackets,
# fusion) will be aligned TO this image, so the masks stay pixel-perfect.
import os, math, time, torch, numpy as np, cv2
from PIL import Image
from google.colab import files

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| gpu:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "-")
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if DEVICE == "cuda" else 0
KEEP_MODELS = VRAM_GB >= 30
print(f"vram: {VRAM_GB:.0f} GB -> " +
      ("models stay loaded (big gpu)" if KEEP_MODELS else "freeing after each cell (small gpu)"))

up = files.upload()
if not up:
    raise RuntimeError("no file uploaded — run this cell again and pick the darker image")
_name = list(up.keys())[0]
darkimg = cv2.imread(_name)
if darkimg is None:
    raise RuntimeError(f"could not read '{_name}' — upload a JPG/PNG")
W = 2000
h, w = darkimg.shape[:2]
if w != W:
    darkimg = cv2.resize(darkimg, (W, int(h * W / w)), interpolation=cv2.INTER_AREA)
H, Wd = darkimg.shape[:2]

# gamma-LIFTED copy (models must SEE) + see-through physics of the ORIGINAL
det = (np.clip((darkimg.astype(np.float32) / 255) ** 0.5, 0, 1) * 255).astype("uint8")
d = darkimg.astype(np.float32) / 255
dl = d.mean(2); dc = d.max(2) - d.min(2)
thr = max(0.22, 0.45 * float(np.percentile(dl, 99.9)))
SEE = (dl >= thr) | (dc >= 0.15)                 # the view: bright OR colourful

GLASS = {}                                       # GLASS["sam3"] / GLASS["oneformer"]
cv2.imwrite("dark_input.jpg", darkimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"darker image: {_name} -> {Wd}x{H} (mean {darkimg.mean():.0f}/255)")
try:
    from IPython.display import Image as _Img, display
    print("THIS image gets segmented AND supplies the window view:")
    display(_Img("dark_input.jpg"))
except Exception:
    pass


def refine_glass(raw):
    """Physics cleanup on the ORIGINAL dark image (same for both models):
    remove thin DARK colourless frame/mullion bars (lit venetian slats KEPT),
    keep thick shaded patches, drop specks."""
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


def show_glass(tag, colour, raw, glass):
    """Overlay on the lifted copy: faint = dropped by refine, solid = final glass."""
    base = det.astype(np.float32)
    for msk, a in ((raw & ~glass, 0.25), (glass, 0.55)):
        al = msk.astype(np.float32)[..., None] * a
        t = np.zeros_like(base); t[:] = colour
        base = base * (1 - al) + t * al
    o = base.clip(0, 255).astype("uint8")
    cnts, _ = cv2.findContours(glass.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cnts, -1, colour, 2)
    cv2.rectangle(o, (0, 0), (o.shape[1] - 1, 44), (0, 0, 0), -1)
    cv2.putText(o, f"{tag} - glass {glass.mean()*100:.1f}%", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    fn = f"glass_{tag.lower().replace(' ', '')}.jpg"
    cv2.imwrite(fn, o, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{tag}: glass = {glass.mean()*100:.1f}% of image")
    try:
        from IPython.display import Image as _Img, display
        display(_Img(fn))
    except Exception:
        pass

print("ready — run CELL 3 (SAM 3) and CELL 4 (OneFormer)")


# ============================== CELL 3 — segment with SAM 3 ===================
# facebook/sam3 is a GATED repo (free): accept the license once at
# https://huggingface.co/facebook/sam3, make a READ token, and add it as the
# Colab secret HF_TOKEN (key icon, "Notebook access" ON) — or paste it when asked.
from huggingface_hub import login
try:
    from google.colab import userdata
    _tok = userdata.get("HF_TOKEN")
except Exception:
    _tok = os.environ.get("HF_TOKEN")
login(token=_tok) if _tok else login()

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
# windows: every instance must GLOW in the dark bracket (doors/walls rejected)
raw = np.zeros((H, Wd), bool)
kept = drop = 0
for mi in sam3_instances(det, ("window pane", "window", "glass door")):
    if float(SEE[mi].mean()) >= 0.06:
        raw |= mi; kept += 1
    else:
        drop += 1
# curtains: rod-hung components only, subtracted
curt_raw = np.zeros((H, Wd), bool)
for mi in sam3_instances(det, ("curtain", "drape")):
    curt_raw |= mi
ncc, lab, st, _ = cv2.connectedComponentsWithStats(curt_raw.astype(np.uint8))
for i in range(1, ncc):
    if st[i, cv2.CC_STAT_TOP] < 0.45 * H and st[i, cv2.CC_STAT_AREA] > 0.0005 * H * Wd:
        raw &= ~(lab == i)
print(f"SAM 3: {kept} window instances kept, {drop} dark rejected  [{time.time()-t0:.1f}s]")

GLASS["sam3"] = refine_glass(raw)
show_glass("SAM3", (255, 150, 0), raw, GLASS["sam3"])

if not KEEP_MODELS:
    del s3_model, s3_proc
    import gc; gc.collect(); torch.cuda.empty_cache()


# ============================== CELL 4 — segment with OneFormer ===============
# ADE20K class 8 "windowpane" (+ class 14 door where light shows through).
# Curtains have their own ADE20K class, so OneFormer excludes them natively.
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
pil = Image.fromarray(cv2.cvtColor(det, cv2.COLOR_BGR2RGB))
inp = of_proc(images=pil, task_inputs=["semantic"], return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = of_model(**inp)
seg = of_proc.post_process_semantic_segmentation(out, target_sizes=[pil.size[::-1]])[0]
seg = seg.cpu().numpy()
raw = (seg == 8) | ((seg == 14) & SEE)           # windowpane + glass door (lit only)
print(f"OneFormer: windowpane raw = {raw.mean()*100:.1f}%  [{time.time()-t0:.1f}s]")

GLASS["oneformer"] = refine_glass(raw)
show_glass("OneFormer", (200, 0, 200), raw, GLASS["oneformer"])

if not KEEP_MODELS:
    del of_model, of_proc
    import gc; gc.collect(); torch.cuda.empty_cache()


# ============================== CELL 5 — upload 7 BRACKETS, align, FUSE =======
# Select ALL exposure brackets of the scene (e.g. 7 JPGs) in one go. Every
# bracket is ECC-aligned TO THE DARKER IMAGE so the masks line up exactly.
print(">>> Upload the exposure BRACKETS now (all together):")
bup = files.upload()
bpaths = sorted(bup.keys())
if len(bpaths) < 2:
    raise RuntimeError("upload at least 2 brackets (usually 5-8)")

bimgs = []
for p in bpaths:
    im = cv2.imread(p)
    if im is None:
        continue
    hh, ww = im.shape[:2]
    if ww != W:
        im = cv2.resize(im, (W, int(hh * W / ww)), interpolation=cv2.INTER_AREA)
    bimgs.append(im)
bimgs.sort(key=lambda im: im.mean())             # darkest -> brightest

# align every bracket to the DARKER image (the masks' coordinate frame)
s = min(1.0, 1000 / W)
g_ref = cv2.equalizeHist(cv2.resize(cv2.cvtColor(darkimg, cv2.COLOR_BGR2GRAY), None,
                                    fx=s, fy=s, interpolation=cv2.INTER_AREA))
S = np.diag([s, s, 1.0])
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
aligned = []
for im in bimgs:
    g_im = cv2.equalizeHist(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None,
                                       fx=s, fy=s, interpolation=cv2.INTER_AREA))
    warp = np.eye(3, dtype=np.float32)
    try:
        cv2.findTransformECC(g_ref, g_im, warp, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
        Hm = (np.linalg.inv(S) @ warp.astype(np.float64) @ S).astype(np.float32)
        aligned.append(cv2.warpPerspective(im, Hm, (Wd, H),
                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_REPLICATE))
    except cv2.error:
        aligned.append(im)

fused = (cv2.createMergeMertens().process(aligned) * 255).clip(0, 255).astype("uint8")
cv2.imwrite("fused.jpg", fused, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"{len(aligned)} brackets (aligned to the darker image) -> fused {Wd}x{H}")
try:
    from IPython.display import Image as _Img, display
    print("FUSED (Mertens):")
    display(_Img("fused.jpg"))
except Exception:
    pass


# ============================== CELL 6 — PICK a segmentation, then HDR ========
# Both masks are shown again; type 1 or 2 to choose. The chosen glass area is
# LEFT ALONE (crisp view from your darker image); HDR applies everywhere else.
# one COMPACT side-by-side strip (not two full-size images) — otherwise the
# input box lands far below the pictures and looks like it never appeared
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

# ---------------------------- CHOOSE HERE ------------------------------------
# Colab shows a DROPDOWN on the right side of this cell (form field). Pick your
# model there — or simply edit the value below — then run this cell.
# No input() box: Colab often fails to render those.
CHOICE = "1"  #@param ["1", "2"] {type:"string"}
# 1 = SAM 3   |   2 = OneFormer
choice = "1" if str(CHOICE).strip() != "2" else "2"
key = "sam3" if choice == "1" else "oneformer"
print(f"-> continuing with {key.upper()}  (glass {GLASS[key].mean()*100:.1f}%)")
print("   (want the other one? change the dropdown/CHOICE value and re-run this cell)")

# EDGE-AWARE feather (guided filter): boundary snaps to the real glass edge in
# the fused image, then the view can blend without a hard cut line.
def _feather(mask, guide_bgr):
    r = max(6, int(mask.shape[1] * 0.004))
    try:
        gd = cv2.cvtColor((np.clip(guide_bgr, 0, 1) * 255).astype(np.uint8),
                          cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        return np.clip(cv2.ximgproc.guidedFilter(gd, mask.astype(np.float32), r, 1e-3), 0, 1)
    except Exception:
        return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), r * 0.6), 0, 1)

glass_view = _feather(GLASS[key].astype(np.float32), fused.astype(np.float32) / 255)

# ---- HDR finishing on the indoor (glass excluded from every measurement) ----
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

def _expose(img, target=0.70):
    m = max(float(np.median(img.mean(2))), 1e-3)
    return img if m >= target else np.clip(img, 0, 1) ** float(np.clip(np.log(target)/np.log(m), 0.45, 1.0))

def _neutralize(img):
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]/255; a = lab[..., 1]-128; b = lab[..., 2]-128
    ch = np.sqrt(a*a+b*b)
    wgt = np.clip((L-0.70)/0.12, 0, 1) * np.clip((22-ch)/8, 0, 1)
    wgt = np.maximum(wgt, np.clip((L-0.85)/0.10, 0, 1))
    wgt = cv2.GaussianBlur(wgt, (0, 0), 8)
    lab[..., 1] = a*(1-wgt)+128; lab[..., 2] = b*(1-wgt)+128
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255

def _match_white(img, strength=0.85):
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    br = lab[..., 0] > 0.78*255
    if br.sum() < 1000: return img
    lab[..., 1] += (128-float(lab[..., 1][br].mean()))*strength
    lab[..., 2] += (128-float(lab[..., 2][br].mean()))*strength
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255

def _scurve(img, s2=0.05): return np.clip(img + s2*np.sin(2*np.pi*(img-0.5)), 0, 1)

def _sharpen(img, fine=0.9, clarity=0.2):
    img = np.clip(img + fine*(img - cv2.GaussianBlur(img, (0, 0), 1.0)), 0, 1)
    wide = cv2.GaussianBlur(img, (0, 0), 15)
    return np.clip(img + clarity*(img - wide), 0, 1)

img = fused.astype(np.float32)/255
ex = (glass_view > 0.3).astype(np.float32)
img = _wb(img, exclude=ex)
img = _levels(img, exclude=ex)
img = _expose(img, 0.70)
img = _neutralize(img)
img = _match_white(img)
img = _scurve(img, 0.05)
img = _sharpen(img)

# ---- glass area is LEFT ALONE: the view comes from YOUR darker image,
#      gently lifted only if it sits too dark next to the enhanced room
vsrc = darkimg.astype(np.float32)/255
sel = glass_view > 0.5
med = float(np.median(vsrc.mean(2)[sel])) if sel.any() else 0.0
if 1e-3 < med < 0.58:
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
print(f"done — segmentation: {key.upper()}; indoor HDR-finished; glass untouched (crisp view)")


# ============================== CELL 7 — OPTIONAL: WHITE FRAMES via Gemini ====
# Sends hdr_result.jpg to gemini-3-pro-image-preview (Nano Banana Pro) asking it
# to recolor the window FRAME to white. Hallucination is blocked TWO ways:
#   1) the prompt forbids any other change + temperature 0 (deterministic)
#   2) HARD GUARANTEE: only the FRAME-BAND pixels are taken from Gemini's
#      output — the glass (your real view), curtains and the whole room are
#      composited back from YOUR result, so nothing else CAN change. Period.
# Cost: ~$0.134 per call. Needs (run once in a cell):
#   !pip install -q -U google-genai
# and a GEMINI_API_KEY from https://aistudio.google.com/apikey — add it as a
# Colab secret (key icon, "Notebook access" ON) or paste it when asked.
from google import genai
from google.genai import types as gtypes

try:
    from google.colab import userdata
    _gkey = userdata.get("GEMINI_API_KEY")
except Exception:
    _gkey = os.environ.get("GEMINI_API_KEY")
if not _gkey:
    _gkey = input("paste your Gemini API key: ")
# strip ALL whitespace/newlines — a key pasted with a line break inside becomes
# an "Illegal header value" and every request fails before it is sent
_gkey = "".join(str(_gkey).split())
gclient = genai.Client(api_key=_gkey)

PROMPT = ("Do not hallucinate anything. Just change the color of the window frame to white. "
          "Keep blinds, curtains, and everything else exactly as they are. "
          "The window style must stay exactly the same. "
          "The outside view through the window must stay exactly the same, with no change "
          "in the colors of the outdoor view. "
          "If there is no frame in the middle of the glass, do not add frames. "
          "Only recolor the existing frame borders to white. Change nothing else in the image.")

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
except Exception as e:                                   # SDK/config version quirks
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

# ---- HARD anti-hallucination composite: Gemini pixels ONLY in the frame band —
# a thin ring around each glass pane + the dark mullion bars near the window.
# Everything else (glass view, curtains, room) is YOUR result, untouched.
g8 = (glass_view > 0.5).astype(np.uint8)
k = max(9, int(Wd * 0.012) | 1)
ring = cv2.dilate(g8, np.ones((k, k), np.uint8)).astype(bool) & ~g8.astype(bool)
near = cv2.dilate(g8, np.ones((k * 4 | 1, k * 4 | 1), np.uint8)).astype(bool)
bars = (dc < 0.07) & (dl < 0.18) & near                  # mullions in/around the window
FRAMEB = cv2.GaussianBlur((ring | bars).astype(np.float32), (0, 0), 2)
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
