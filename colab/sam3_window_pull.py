"""
SAM 3 WINDOW PULL — full HDR pipeline with Meta SAM 3 as the segmentation engine.

Flow (exactly as requested):
  CELL 3  you upload the exposure BRACKETS (e.g. 7 JPGs of one scene)
          -> they are aligned (ECC homography) and FUSED (Mertens)
  CELL 4  you upload the DARKER / VIEW image (the one whose window view you want)
          -> aligned to the brackets
  CELL 5  SAM 3 segments the WINDOW GLASS PANE area (text -> instance masks),
          then CHECKS & REMOVES curtains (SAM 3 on the fused image, rod-hung
          geometry) and the frame/mullions (dark-image physics) — glass only
  CELL 6  indoor gets the full HDR finishing; the glass gets NO HDR — the view
          is composited crisp from YOUR darker image. Results downloaded.

Model: facebook/sam3 (gated but free — one-time license accept + HF token).
Segmentation lessons baked in (from the 8-model shootout on 3T1A3833):
  * SAM 3 looks at a gamma-LIFTED copy of the dark image AND at the fused image
    (union) — a dark bracket alone hides windows from any model
  * every window instance must GLOW in the dark bracket (see-through gate) —
    a dark box is a door/wall, never a window with a view
  * frame bars must be DARK colourless thin lines — bright thin lines are
    venetian-blind slats / lit view and are KEPT
  * curtains come from SAM 3 "curtain"/"drape" on the FUSED image, kept only if
    rod-hung (component top in the upper part of the image), then subtracted

Run on Google Colab with a GPU (A100 or T4). Copy each CELL into its own cell.
"""

# ============================== CELL 1 — install ==============================
# (Runtime -> Change runtime type -> GPU, first!)
# !pip install -q -U transformers opencv-python-headless
# !pip install -q --force-reinstall "pillow==10.4.0"
# !pip install -q --force-reinstall -U huggingface_hub
#
# >>> AFTER THIS CELL: Runtime -> Restart session, then run CELL 2. <<<


# ============================== CELL 2 — HF login + load SAM 3 ================
# facebook/sam3 is a GATED repo (free, but Meta's license must be accepted once):
#   1) log in at huggingface.co and open https://huggingface.co/facebook/sam3
#      -> click "Agree and access repository" (check huggingface.co/settings/
#      gated-repos until it says "Accepted")
#   2) make a READ token at https://huggingface.co/settings/tokens
#   3) in Colab: left sidebar key icon (Secrets) -> add HF_TOKEN = your token,
#      "Notebook access" ON — or run this cell and paste the token when asked.
import os, math, time, torch, numpy as np, cv2
from PIL import Image
from huggingface_hub import login

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| gpu:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "-")

try:
    from google.colab import userdata
    _tok = userdata.get("HF_TOKEN")
except Exception:
    _tok = os.environ.get("HF_TOKEN")
login(token=_tok) if _tok else login()

from transformers import Sam3Processor, Sam3Model

s3_proc = Sam3Processor.from_pretrained("facebook/sam3")
s3_model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE).eval()
print("SAM 3 loaded")


def sam3_instances(bgr, phrases, threshold=0.35):
    """SAM 3 text-prompted instances: list of bool masks (union over phrases,
    every instance kept separate so gates can judge each one)."""
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


# ============================== CELL 3 — upload BRACKETS, align, FUSE =========
# Select ALL exposure brackets of ONE scene (e.g. the 7 JPGs), in one go.
from google.colab import files

up = files.upload()
paths = sorted(up.keys())
if len(paths) < 2:
    raise RuntimeError("upload at least 2 brackets (usually 5-8)")


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
items.sort(key=lambda it: it["img"].mean())              # darkest -> brightest
res = [it["img"] for it in items]

times = [it["t"] for it in items if it["t"]]
med_t = sorted(times)[len(times) // 2] if times else None
EVS = []
for it in items:
    ev = round(math.log2(it["t"] / med_t), 1) if (it["t"] and med_t) else None
    EVS.append(ev)
print("brackets (darkest -> brightest):",
      [f"#{i} EV{e:+g}" if e is not None else f"#{i} ?" for i, e in enumerate(EVS)])


# ALIGN every bracket to the middle one (ECC homography) BEFORE fusing — without
# this the pull composites slightly-shifted brackets and the view goes milky.
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
            Hm = (np.linalg.inv(S) @ warp.astype(np.float64) @ S).astype(np.float32)
            out.append(cv2.warpPerspective(im, Hm, (w, h),
                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_REPLICATE))
        except cv2.error:
            out.append(im)
    return out


res = _align(res)
fused = (cv2.createMergeMertens().process(res) * 255).clip(0, 255).astype("uint8")
cv2.imwrite("fused.jpg", fused, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"{len(res)} brackets (aligned) -> fused {fused.shape[1]}x{fused.shape[0]}")
try:
    from IPython.display import Image as _Img, display
    print("FUSED (Mertens, aligned brackets):")
    display(_Img("fused.jpg"))
except Exception:
    pass


# ============================== CELL 4 — upload the DARKER / VIEW image =======
# Upload the ONE darker image whose window view you want composited into the
# glass. It is resized and ECC-aligned to the same reference as the brackets.
print(">>> Upload the DARKER / VIEW image now:")
vup = files.upload()
vpath = list(vup.keys())[0]
viewimg = cv2.imread(vpath)
vh, vw = viewimg.shape[:2]
if vw != W:
    viewimg = cv2.resize(viewimg, (W, int(vh * W / vw)), interpolation=cv2.INTER_AREA)

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

# gamma-LIFTED copy (models must SEE) + see-through physics of the ORIGINAL
det = (np.clip((viewimg.astype(np.float32) / 255) ** 0.5, 0, 1) * 255).astype("uint8")
d = viewimg.astype(np.float32) / 255
dl = d.mean(2); dc = d.max(2) - d.min(2)
thr = max(0.22, 0.45 * float(np.percentile(dl, 99.9)))
SEE = (dl >= thr) | (dc >= 0.15)                 # the view: bright OR colourful

cv2.imwrite("view_bracket.jpg", viewimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"view/darker image: {vpath} (mean {viewimg.mean():.0f}/255)")
try:
    from IPython.display import Image as _Img, display
    print("THIS image supplies the window view:")
    display(_Img("view_bracket.jpg"))
except Exception:
    pass


# ============================== CELL 5 — SAM 3: GLASS minus curtains/frame ====
H, Wd = fused.shape[:2]
t0 = time.time()

# ---- 1) WINDOWS: SAM 3 on the lifted DARK image AND on the FUSED image (union
#         of both runs = recall), every instance gated by "does it glow in the
#         dark bracket?" — doors/walls/mirrors have no light inside and drop out.
WIN = np.zeros((H, Wd), bool)
n_kept = n_drop = 0
for src in (det, fused):
    for mi in sam3_instances(src, ("window pane", "window", "glass door")):
        if float(SEE[mi].mean()) >= 0.06:
            WIN |= mi; n_kept += 1
        else:
            n_drop += 1
print(f"windows: {n_kept} instances kept, {n_drop} dark ones rejected")

# small erode lip so the OUTER frame edge is excluded from the start
lip = max(3, int(Wd * 0.004) | 1)
WIN_IN = cv2.erode(WIN.astype(np.uint8), np.ones((lip, lip), np.uint8)).astype(bool)

# ---- 2) CURTAINS: SAM 3 on the FUSED image (curtains are visible there), kept
#         only if ROD-HUNG (top of the component in the upper part of the image)
#         — a pool umbrella outside mislabelled 'curtain' is a mid-window blob.
curt_raw = np.zeros((H, Wd), bool)
for mi in sam3_instances(fused, ("curtain", "drape")):
    curt_raw |= mi
ncc, lab, st, _ = cv2.connectedComponentsWithStats(curt_raw.astype(np.uint8))
CURT = np.zeros((H, Wd), bool)
for i in range(1, ncc):
    if st[i, cv2.CC_STAT_TOP] < 0.45 * H and st[i, cv2.CC_STAT_AREA] > 0.0005 * H * Wd:
        CURT |= (lab == i)
print(f"curtains: {CURT.mean()*100:.1f}% of image (rod-hung only)")

# ---- 3) FRAME / MULLIONS: thin DARK colourless lines in the dark image. Bright
#         thin lines are venetian-blind slats / lit view — KEPT (the shootout
#         lesson: an any-brightness rule eats entire blinds).
achro_dark = (dc < 0.07) & (dl < 0.18) & WIN_IN
bar_k = max(5, int(Wd * 0.014) | 1)
blobs = cv2.morphologyEx(achro_dark.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((bar_k, bar_k), np.uint8)).astype(bool)
FRAME = achro_dark & ~blobs                      # thin dark bars; thick dark = shade, kept
print(f"frame bars: {FRAME.mean()*100:.1f}% of image")

# ---- GLASS = window interior minus curtains minus frame ----
glass = WIN_IN & ~CURT & ~FRAME
glass = cv2.morphologyEx(glass.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
ncc, lab, st, _ = cv2.connectedComponentsWithStats(glass.astype(np.uint8))
for i in range(1, ncc):                          # a real pane is never a speck
    if st[i, cv2.CC_STAT_AREA] < 0.0005 * H * Wd:
        glass[lab == i] = False

# EDGE-AWARE feather (guided filter): boundary snaps to the real glass edge in
# the fused image; clamped so it can never leave the window area.
def _feather(mask, guide_bgr):
    r = max(6, int(mask.shape[1] * 0.004))
    try:
        gd = cv2.cvtColor((np.clip(guide_bgr, 0, 1) * 255).astype(np.uint8),
                          cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        return np.clip(cv2.ximgproc.guidedFilter(gd, mask.astype(np.float32), r, 1e-3), 0, 1)
    except Exception:
        return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), r * 0.6), 0, 1)

glass_view = _feather(glass.astype(np.float32), fused.astype(np.float32) / 255)
glass_view = glass_view * WIN.astype(np.float32)
print(f"GLASS (view) = {(glass_view > 0.5).mean()*100:.1f}% of image   [{time.time()-t0:.1f}s]")

# ---- show the verified segments: CYAN glass · ORANGE frame · MAGENTA curtains
seg_vis = fused.astype(np.float32)
for msk, col in (((glass_view > 0.5), (220, 220, 0)), (FRAME, (0, 140, 255)), (CURT, (200, 0, 200))):
    a = msk.astype(np.float32)[..., None] * 0.5
    t = np.zeros_like(seg_vis); t[:] = col
    seg_vis = seg_vis * (1 - a) + t * a
cv2.imwrite("segments_verified.jpg", seg_vis.clip(0, 255).astype("uint8"),
            [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite("glass_mask.png", ((glass_view > 0.5).astype(np.uint8) * 255))
try:
    from IPython.display import Image as _Img, display
    print("VERIFIED segments — cyan glass · orange frame · magenta curtains:")
    display(_Img("segments_verified.jpg"))
except Exception:
    pass


# ============================== CELL 6 — HDR finishing + WINDOW PULL ==========
# Indoor (room, curtains, frame) gets the full finishing; the GLASS gets NO HDR —
# the view is composited crisp from YOUR darker image (CELL 4).
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

def _neutralize(img):
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]/255; a = lab[..., 1]-128; b = lab[..., 2]-128
    ch = np.sqrt(a*a+b*b)
    w = np.clip((L-0.70)/0.12, 0, 1) * np.clip((22-ch)/8, 0, 1)
    w = np.maximum(w, np.clip((L-0.85)/0.10, 0, 1))
    w = cv2.GaussianBlur(w, (0, 0), 8)
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

# ---- OUTDOOR VIEW = the darker image YOU uploaded, gently lifted if it sits dark
vsrc = viewimg.astype(np.float32)/255
sel = glass_view > 0.5
med = float(np.median(vsrc.mean(2)[sel])) if sel.any() else 0.0
if 1e-3 < med < 0.58:
    vsrc = np.clip(vsrc, 0, 1) ** float(np.clip(np.log(0.58)/np.log(med), 0.6, 1.0))
gv = glass_view[..., None]
img = img*(1-gv) + vsrc*gv

result = (img*255).clip(0, 255).astype("uint8")
cv2.imwrite("hdr_result.jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
try:
    from IPython.display import Image as _Img, display
    print("FUSED (base going in):"); display(_Img("fused.jpg"))
    print("FINAL — indoor enhanced + crisp outdoor view from your darker image:")
    display(_Img("hdr_result.jpg"))
except Exception:
    pass
files.download("fused.jpg")
files.download("segments_verified.jpg")
files.download("glass_mask.png")
files.download("hdr_result.jpg")
print("done — SAM 3 glass pull: indoor enhanced; glass view from your darker image")
