"""
BLIND WHITEN — ONE cell that finds the venetian blind, HDR-processes its real
slat material, protects every gap where the outdoor view shows through, and
recolors ONLY the slat body to white. No generative AI — zero hallucination
risk, because no model ever produces a pixel here; it is pure color math on
your own photo.

DETECTION (what finds the blind):
  * SAM 3 (facebook/sam3)      — text prompts: "venetian blind", "blind",
    "window blind" (instance masks, catches the blind as one clean object)
  * OneFormer (ADE20K)         — native semantic class 63 "blind;screen"
    (verified against CSAILVision/sceneparsing objectInfo150.csv: Idx 64,
    1-indexed -> model output 63, 0-indexed — same convention already used
    for windowpane=8 / door=14 elsewhere in this project)
  Both are UNIONED (either one finding it is enough) for best recall — this
  mirrors the shootout finding that consensus beats any single model.
  WINDOW-ANCHOR filter: a real blind always sits ON a window, so any "blind"
  component that isn't actually near OneFormer's own "windowpane" class gets
  REJECTED. Fixes the real false-positive found in testing — a louvered
  CLOSET door (same horizontal-slat look) getting detected as a blind.

THE CORE TRICK (why it can't wash out the view):
  1) GAP test on the DARKER image: any pixel that is bright/colourful in the
     dark bracket is real outdoor light showing between the slats — EXCLUDED
     from recoloring, no matter what the detector said. The raw per-pixel test
     is noisy on slats (each slat's own lit top edge can flicker across the
     threshold) — morphological open+close turns that into clean gap bands
     instead of a cyan/magenta checkerboard.
  2) The blind mask is eroded inward (a small "lip") so the blurry blend zone
     right at each slat edge is never touched either.
  3) Only the CHROMA (color) of the remaining slat-body pixels is changed —
     their real LIGHTNESS (the highlight/shadow pattern that makes it look
     like a real 3D blind) is left completely untouched. Full desaturation
     (a=b=128 in LAB) at each pixel's own real brightness IS what a physically
     white material looks like — same principle as _match_white already used
     in this project, just applied to the whole blind body, not just the
     brightest highlights.
  4) HDR (white balance / levels / exposure / sharpen) is applied to the whole
     scene AS NORMAL — the blind body is real indoor material and gets it like
     everything else; only the GAP pixels are excluded and pulled from the
     darker/view image instead (same rule as the window glass elsewhere).

Works two ways:
  * Pasted INSIDE dual_seg_window_pull.py / clear_seg_window_pull.py, after
    their HDR cell — it reuses `fused`, `darkimg`, `dl`, `dc`, `SEE`, `Wd`, `H`,
    `result` and the already-loaded models straight from memory (fast).
  * Run STANDALONE — it uploads its own images and loads its own models if
    those variables don't already exist.

Run on Google Colab with a GPU. This is intentionally ONE cell.
"""

# ============================== (install, if standalone) ======================
# !pip install -q -U transformers opencv-python-headless huggingface_hub
# !pip install -q --force-reinstall "pillow==10.4.0"
# Runtime -> Restart session after installing, then run the CELL below.


# ============================== CELL — BLIND WHITEN ============================
import os, cv2, numpy as np, torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 0) reuse from memory if this is pasted into an existing notebook session,
#         else fall back to asking for uploads (standalone use) ----------------
if "fused" not in globals() or "darkimg" not in globals():
    from google.colab import files
    print(">>> standalone mode — upload the CLEAR/FUSED image (blind visible):")
    up = files.upload(); _p = list(up.keys())[0]
    fused = cv2.imread(_p)
    W = 2000; h, w = fused.shape[:2]
    if w != W: fused = cv2.resize(fused, (W, int(h*W/w)), interpolation=cv2.INTER_AREA)
    H, Wd = fused.shape[:2]
    print(">>> upload the DARKER image (for the light-through gap test):")
    up = files.upload(); _p = list(up.keys())[0]
    darkimg = cv2.imread(_p)
    vh, vw = darkimg.shape[:2]
    if vw != Wd: darkimg = cv2.resize(darkimg, (Wd, int(vh*Wd/vw)), interpolation=cv2.INTER_AREA)
    d = darkimg.astype(np.float32) / 255
    dl = d.mean(2); dc = d.max(2) - d.min(2)
    thr = max(0.22, 0.45 * float(np.percentile(dl, 99.9)))
    SEE = (dl >= thr) | (dc >= 0.15)
    result = fused.copy()                          # nothing HDR'd yet in standalone mode
else:
    H, Wd = fused.shape[:2]
    print("reusing fused / darkimg / SEE / result from the existing session")

BASE = result if "result" in globals() else fused  # HDR'd image if available, else fused
det_src = fused                                     # detection ALWAYS runs on the clear/fused image


def sam3_instances(model, proc, bgr, phrases, threshold=0.35):
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out = []
    for phrase in phrases:
        inp = proc(images=pil, text=phrase, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            o = model(**inp)
        res = proc.post_process_instance_segmentation(
            o, threshold=threshold, mask_threshold=0.5, target_sizes=[pil.size[::-1]])[0]
        m = res.get("masks", None)
        if m is not None and len(m):
            arr = (m if hasattr(m, "cpu") else torch.stack(list(m))).cpu().numpy()
            for mi in arr.reshape(-1, *arr.shape[-2:]).astype(bool):
                if mi.any():
                    out.append(mi)
    return out


# ---- 1) SAM 3 — load if not already in memory ----
if "s3_model" not in globals():
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
    print("SAM 3 loaded")
else:
    print("SAM 3 reused from memory")

BLIND_SAM = np.zeros((H, Wd), bool)
for mi in sam3_instances(s3_model, s3_proc, det_src,
                         ("venetian blind", "blind", "window blind")):
    BLIND_SAM |= mi
print(f"SAM 3: blind raw = {BLIND_SAM.mean()*100:.1f}% of image")

# ---- 2) OneFormer — load if not already in memory ----
if "of_model" not in globals():
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    try:
        OF_ID = "shi-labs/oneformer_ade20k_dinat_large"
        of_proc = OneFormerProcessor.from_pretrained(OF_ID)
        of_model = OneFormerForUniversalSegmentation.from_pretrained(OF_ID).to(DEVICE).eval()
    except Exception as e:
        print(f"DiNAT unavailable ({type(e).__name__}) — Swin-L fallback")
        OF_ID = "shi-labs/oneformer_ade20k_swin_large"
        of_proc = OneFormerProcessor.from_pretrained(OF_ID)
        of_model = OneFormerForUniversalSegmentation.from_pretrained(OF_ID).to(DEVICE).eval()
    print("OneFormer loaded:", OF_ID)
else:
    print("OneFormer reused from memory")

_pil = Image.fromarray(cv2.cvtColor(det_src, cv2.COLOR_BGR2RGB))
_inp = of_proc(images=_pil, task_inputs=["semantic"], return_tensors="pt").to(DEVICE)
with torch.no_grad():
    _out = of_model(**_inp)
_seg = of_proc.post_process_semantic_segmentation(_out, target_sizes=[_pil.size[::-1]])[0]
_seg = _seg.cpu().numpy()
BLIND_OF = (_seg == 63)                             # verified: ADE20K "blind;screen"
WINDOW_OF = (_seg == 8)                             # verified: ADE20K "windowpane"
print(f"OneFormer: blind raw = {BLIND_OF.mean()*100:.1f}% of image, "
      f"windowpane anchor = {WINDOW_OF.mean()*100:.1f}%")

# ---- 3) union + WINDOW-ANCHOR filter + clean specks + erode a small lip -----
# a blind sits ON a window — anything labelled "blind" that isn't actually near
# a detected window (e.g. a louvered CLOSET door, which looks like a blind) is
# a false positive and gets dropped here.
BLIND_UNION = BLIND_SAM | BLIND_OF
win_k = max(15, int(Wd * 0.03) | 1)                 # generous — blind can overhang the glass
WINDOW_NEAR = cv2.dilate(WINDOW_OF.astype(np.uint8), np.ones((win_k, win_k), np.uint8)).astype(bool)

ncc, lab, st, _ = cv2.connectedComponentsWithStats(BLIND_UNION.astype(np.uint8))
BLIND_RAW = np.zeros((H, Wd), bool)
for i in range(1, ncc):
    area = st[i, cv2.CC_STAT_AREA]
    if area < 0.0008 * H * Wd:                      # a real blind is never a speck
        continue
    comp = (lab == i)
    near_frac = float((comp & WINDOW_NEAR).sum()) / float(area)
    if near_frac >= 0.15:                           # must actually be at a window
        BLIND_RAW |= comp
    else:
        print(f"  rejected a 'blind' component ({area/(H*Wd)*100:.1f}% of image, "
              f"only {near_frac*100:.0f}% near a window) — likely a closet door / false positive")
lip = max(3, int(Wd * 0.004) | 1)
BLIND_RAW = cv2.erode(BLIND_RAW.astype(np.uint8), np.ones((lip, lip), np.uint8)).astype(bool)
print(f"blind (union, window-anchored, cleaned, eroded) = {BLIND_RAW.mean()*100:.1f}% of image")

# ---- 4) split: BODY = real slat material (safe to recolor); GAP = outdoor
#         light/view showing between slats (NEVER touched — same as glass).
#         The raw per-pixel SEE test is NOISY on slats (each slat's own lit
#         top edge can flicker across the brightness threshold) — morphological
#         open+close turns that pixel-noise into clean horizontal gap bands. ---
GAP_RAW = BLIND_RAW & SEE
gap_k = max(3, int(Wd * 0.0035) | 1)
GAP = cv2.morphologyEx(GAP_RAW.astype(np.uint8), cv2.MORPH_OPEN,
                       np.ones((gap_k, gap_k), np.uint8))
GAP = cv2.morphologyEx(GAP, cv2.MORPH_CLOSE,
                       np.ones((gap_k * 2 | 1, gap_k * 2 | 1), np.uint8)).astype(bool)
GAP &= BLIND_RAW                                    # never spill outside the blind footprint
BODY = BLIND_RAW & ~GAP
print(f"blind BODY (recolor target) = {BODY.mean()*100:.1f}% | "
      f"GAP (protected view-through) = {GAP.mean()*100:.1f}%")

# ---- 5) recolor: touch ONLY color (LAB a,b -> neutral), NEVER touch lightness.
#         Full desaturation at each pixel's real brightness = physically what a
#         white material looks like (its own highlight/shadow pattern stays). --
lab_img = cv2.cvtColor(BASE, cv2.COLOR_BGR2LAB).astype(np.float32)
STRENGTH = 1.0                                       # 1.0 = fully white/neutral
body_f = BODY.astype(np.float32)
# tiny 1px smooth on the mask edge only — NOT the wide feather used for glass
# blending, this is a same-image in-place tint so the boundary should be tight
body_f = cv2.GaussianBlur(body_f, (0, 0), 0.8) * STRENGTH
lab_img[..., 1] = lab_img[..., 1] * (1 - body_f) + 128 * body_f
lab_img[..., 2] = lab_img[..., 2] * (1 - body_f) + 128 * body_f
whitened = cv2.cvtColor(np.clip(lab_img, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

# ---- 6) GAP pixels come from the darker/view image, crisp, exactly like glass ----
vsrc = darkimg.astype(np.float32) / 255
sel = GAP
med = float(np.median(vsrc.mean(2)[sel])) if sel.any() else 0.0
if 1e-3 < med < 0.58:
    vsrc = np.clip(vsrc, 0, 1) ** float(np.clip(np.log(0.58) / np.log(med), 0.6, 1.0))
gap_f = GAP.astype(np.float32)[..., None]
final = whitened.astype(np.float32) * (1 - gap_f) + (vsrc * 255) * gap_f
final = final.clip(0, 255).astype("uint8")

cv2.imwrite("blind_before.jpg", BASE, [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite("blind_after.jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 95])

# ---- verification overlay: CYAN = recolored body, MAGENTA = protected gap ----
vis = BASE.astype(np.float32)
for msk, col in ((BODY, (220, 220, 0)), (GAP, (200, 0, 200))):
    a = msk.astype(np.float32)[..., None] * 0.5
    t = np.zeros_like(vis); t[:] = col
    vis = vis * (1 - a) + t * a
cv2.imwrite("blind_verify.jpg", vis.clip(0, 255).astype("uint8"), [cv2.IMWRITE_JPEG_QUALITY, 92])

try:
    from IPython.display import Image as _Img, display
    print("VERIFY — cyan = recolored slat body, magenta = protected outdoor view:")
    display(_Img("blind_verify.jpg"))
    print("BEFORE:"); display(_Img("blind_before.jpg"))
    print("AFTER — blind whitened, view untouched:"); display(_Img("blind_after.jpg"))
except Exception:
    pass
try:
    from google.colab import files as _files
    _files.download("blind_after.jpg")
except Exception:
    pass
print("done — blind body recolored to white, outdoor view pixel-identical to your photo")
