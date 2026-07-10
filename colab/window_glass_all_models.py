"""
WINDOW GLASS DETECTION — THE FULL 8-MODEL SHOOTOUT on the DARKER exposure.

Upload the darker bracket(s) (where the outdoor view is clear); every pipeline
detects ONLY the see-through glass — the pane area where the outdoor view shows.
All models are OPEN — no API key, nothing paid. (SAM 3 needs a one-time free
HuggingFace login, that's all.)

  cell  pipeline           models used (exact ids)                              license
  ----  -----------------  ---------------------------------------------------  ----------
  C4    A) Grounded SAM 2  IDEA-Research/grounding-dino-base + SAM 2.1 Hiera-L  Apache-2.0
  C5    B) DINO + HQ-SAM   same DINO boxes + lkeab/hq-sam sam_hq_vit_h.pth      Apache-2.0
  C6    C) SAM 3           facebook/sam3 (text -> all instance masks, 1 model)  SAM license*
  C7    D) Florence-2+SAM  microsoft/Florence-2-large (text -> boxes) + SAM2.1  MIT
  C8    E) OneFormer       shi-labs/oneformer_ade20k_dinat_large (58.3 mIoU;    MIT code**
                           auto-fallback: oneformer_ade20k_swin_large)
  C9    F) Mask2Former     facebook/mask2former-swin-large-ade-semantic         CC BY-NC!!
  C10   G) EoMT            tue-mps/ade20k_semantic_eomt_large_512               MIT
                           (CVPR 2025 — DINOv2 ViT does segmentation alone)
  C11   H) SegGPT          BAAI/seggpt-vit-large — IN-CONTEXT: you upload ONE   MIT
                           example (dark image + its white glass mask) and it
                           copies that segmentation onto every new image
  C12   comparison grid + agreement map + consensus mask + zip

  *  gated but free: accept the license once at hf.co/facebook/sam3 + HF token
  ** weights released for research; CC BY-NC = NON-commercial — testing only
  E/F/G find glass via the ADE20K class 8 "windowpane" (+ class 14 door where
  light shows through); A/B/C/D find it via the text prompts window/windowpane/
  glass door. H copies YOUR own marked example.

Fairness rules (same for every pipeline):
  * models LOOK at a gamma-lifted copy (a dark bracket hides windows from any
    model); all masks apply to the ORIGINAL dark image
  * identical glass-refine: thin DARK colourless frame/mullion bars removed,
    lit venetian-blind slats & thick shaded view patches kept, specks dropped
  * detector boxes with no light inside are rejected (a dark box is a door/wall,
    never a window) and curtain masks are subtracted (A/B/D)
  * on a big GPU (A100) every model stays loaded; on a T4 each cell frees its
    model afterwards — decided automatically
  * one darker image or many at once — every cell loops

Run on Google Colab with a GPU (A100 best, T4 works). One CELL per Colab cell.
"""

# ============================== CELL 1 — install ==============================
# (Runtime -> Change runtime type -> GPU : A100 or T4, first!)
# !pip install -q -U transformers ultralytics opencv-python-headless huggingface_hub
# !pip install -q segment-anything-hq timm einops
# !pip install -q --force-reinstall "pillow==10.4.0"
# # natten is OPTIONAL (only for OneFormer's strongest DiNAT backbone; CELL 8
# # falls back to Swin automatically). shi-labs.com's SSL cert is expired and
# # the source build usually fails on Colab — if this line errors, IGNORE it:
# !pip install -q natten -f https://shi-labs.com/natten/wheels --trusted-host shi-labs.com
#
# >>> AFTER THIS CELL: Runtime -> Restart session, then run CELL 2. <<<


# ============================== CELL 2 — upload the DARKER image(s) ===========
# Upload the darker exposure(s) where the window view is clear. One or many.
import os, time, torch, numpy as np, cv2
from PIL import Image
from google.colab import files

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| gpu:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "-")

# A100 (40/80 GB): keep every model loaded -> re-running any cell is instant.
# T4 (16 GB): free each model after its cell so all pipelines fit.
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if DEVICE == "cuda" else 0
KEEP_MODELS = VRAM_GB >= 30
print(f"vram: {VRAM_GB:.0f} GB -> " +
      ("models stay loaded (big gpu)" if KEEP_MODELS else "freeing after each cell (small gpu)"))

up = files.upload()
if not up:
    raise RuntimeError("no file uploaded — run this cell again and pick the darker image(s)")

W = 2000
IMAGES = []                       # [{name, dark(BGR), det(BGR gamma-lifted), see(bool)}]
for _name in sorted(up.keys()):
    im = cv2.imread(_name)
    if im is None:
        print(f"  ! could not read '{_name}' — skipped"); continue
    h, w = im.shape[:2]
    if w != W:
        im = cv2.resize(im, (W, int(h * W / w)), interpolation=cv2.INTER_AREA)
    # models must SEE: a dark bracket hides the dimmer windows from ANY model —
    # they all look at this gamma-lifted copy; masks/physics use the ORIGINAL.
    det = (np.clip((im.astype(np.float32) / 255) ** 0.5, 0, 1) * 255).astype("uint8")
    # see-through = the outdoor view in the dark image: bright OR colourful
    d = im.astype(np.float32) / 255
    dl = d.mean(2); dc = d.max(2) - d.min(2)
    thr = max(0.22, 0.45 * float(np.percentile(dl, 99.9)))
    see = (dl >= thr) | (dc >= 0.15)
    IMAGES.append({"name": os.path.splitext(_name)[0], "dark": im, "det": det, "see": see})
    print(f"  {_name}: {im.shape[1]}x{im.shape[0]} (mean {im.mean():.0f}/255)")

RESULTS = {}                      # RESULTS[pipeline][image_name] = bool glass mask
os.makedirs("out_all", exist_ok=True)
try:
    from IPython.display import Image as _Img, display
    for it in IMAGES:
        cv2.imwrite("_show.jpg", it["dark"], [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"darker input '{it['name']}' (THIS gets segmented):")
        display(_Img("_show.jpg"))
except Exception:
    pass


# ============================== CELL 3 — shared helpers =======================
def free_gpu():
    """Reclaim VRAM (call AFTER `del`-ing the models) so the next pipeline fits
    on the T4. `del` inside a function would only drop a local reference."""
    import gc
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"gpu freed — {torch.cuda.memory_allocated()/1e9:.1f} GB still allocated")


def refine_glass(raw, it):
    """Same cleanup for EVERY pipeline (fair comparison), on the ORIGINAL dark image:
    remove thin DARK colourless frame/mullion bars; KEEP thick colourless patches
    (shaded view) and anything LIT (white venetian-blind slats, bright view lines);
    drop specks.
    LESSON from the 3T1A3833 test: bars must be DARK (dl < 0.18) — the earlier
    any-brightness rule ate entire venetian blinds (white slats = thin colourless
    lines) and left 0.3% of a correct raw mask. A real frame is unlit in the dark
    bracket; anything bright inside the window IS the lit opening — keep it."""
    H, Wd = raw.shape
    d = it["dark"].astype(np.float32) / 255
    dl = d.mean(2); dc = d.max(2) - d.min(2)
    m = raw.astype(bool).copy()
    achro = (dc < 0.07) & (dl < 0.18) & m                # DARK colourless pixels only
    bar_k = max(5, int(Wd * 0.014) | 1)
    blobs = cv2.morphologyEx(achro.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((bar_k, bar_k), np.uint8)).astype(bool)
    m &= ~(achro & ~blobs)                               # thin dark bars out, blobs stay
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    ncc, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    for i in range(1, ncc):                              # a real pane is never a speck
        if st[i, cv2.CC_STAT_AREA] < 0.0005 * H * Wd:
            m[lab == i] = False
    return m


def see_frac(box, it):
    """Fraction of see-through (lit) pixels inside a detector box, on the ORIGINAL
    dark image. A real window ALWAYS glows in the dark bracket; a closet/wooden
    door does not — this gate killed the 'white closet door = glass door' bug."""
    x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
    x2 = min(int(box[2]), it["see"].shape[1]); y2 = min(int(box[3]), it["see"].shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(it["see"][y1:y2, x1:x2].mean())


def show_pipeline(tag, colour, it, raw, glass):
    """Overlay (on the lifted copy, so you can see) + binary mask, saved and shown.
    Faint tint = dropped by the refine; solid tint + contour = final glass."""
    base = it["det"].astype(np.float32)
    for msk, a in ((raw & ~glass, 0.25), (glass, 0.55)):
        al = msk.astype(np.float32)[..., None] * a
        t = np.zeros_like(base); t[:] = colour
        base = base * (1 - al) + t * al
    o = base.clip(0, 255).astype("uint8")
    cnts, _ = cv2.findContours(glass.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cnts, -1, colour, 2)
    cv2.rectangle(o, (0, 0), (o.shape[1] - 1, 44), (0, 0, 0), -1)
    cv2.putText(o, f"{tag} - {it['name']} - glass {glass.mean()*100:.1f}%", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    slug = tag.split()[0].lower()                        # "A GroundedSAM2" -> "a"
    cv2.imwrite(f"out_all/{it['name']}_{slug}_overlay.jpg", o, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(f"out_all/{it['name']}_{slug}_mask.png", glass.astype(np.uint8) * 255)
    print(f"  {tag:16s} {it['name']}: raw {raw.mean()*100:5.1f}% -> glass {glass.mean()*100:5.1f}%")
    try:
        from IPython.display import Image as _Img, display
        display(_Img(f"out_all/{it['name']}_{slug}_overlay.jpg"))
    except Exception:
        pass


def load_dino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    GD_ID = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(GD_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_ID).to(DEVICE).eval()
    return proc, model


def dino_boxes(proc, model, it, phrases=("window", "windowpane", "glass door"),
               gate=0.06, box_t=0.25, text_t=0.20):
    """Grounding DINO synonym ensemble — union of boxes over several phrasings.
    Boxes with no light inside (see_frac < gate) are REJECTED: that's a door or
    a wall, never a window with a view. gate=0.0 disables (curtain detection)."""
    pil = Image.fromarray(cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB))
    allb = []
    for phrase in phrases:
        inp = proc(images=pil, text=phrase + " .", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        try:
            det = proc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=box_t, text_threshold=text_t,
                target_sizes=[pil.size[::-1]])[0]
        except TypeError:
            det = proc.post_process_grounded_object_detection(
                out, inp.input_ids, box_threshold=box_t, text_threshold=text_t,
                target_sizes=[pil.size[::-1]])[0]
        if len(det["boxes"]):
            allb.append(det["boxes"].cpu().numpy())
    boxes = np.vstack(allb) if allb else np.zeros((0, 4))
    if gate > 0 and len(boxes):
        keep = np.array([see_frac(b, it) >= gate for b in boxes])
        if (~keep).sum():
            print(f"    {int((~keep).sum())} dark box(es) rejected (no light inside)")
        boxes = boxes[keep]
    return boxes


def sam2_union(sam2, bgr, boxes):
    """SAM 2.1: union of pixel-precise masks for the given boxes."""
    if len(boxes) == 0:
        return np.zeros(bgr.shape[:2], bool)
    try:
        r = sam2(bgr, bboxes=boxes.tolist(), retina_masks=True, verbose=False)
    except TypeError:
        r = sam2(bgr, bboxes=boxes.tolist(), verbose=False)
    if not r or r[0].masks is None:
        return np.zeros(bgr.shape[:2], bool)
    return r[0].masks.data.cpu().numpy().astype(bool).any(axis=0)


def ade_windowpane(seg_map, it):
    """ADE20K semantics -> glass: 'windowpane'(8) everywhere; 'door'(14) only where
    light actually shows through (a glass door passes, a solid wood door is dark)."""
    return (seg_map == 8) | ((seg_map == 14) & it["see"])

print("helpers ready")


# ============================== CELL 4 — A: Grounded SAM 2 ====================
# Grounding DINO base (text -> boxes) + SAM 2.1 Hiera-Large (boxes -> masks).
# The proven champion pipeline (already used in this repo).      [Apache-2.0]
from ultralytics import SAM

gd_proc, gd_model = load_dino()
sam2 = SAM("sam2.1_l.pt")                                # auto-downloads the checkpoint
DINO_BOXES = {}                                          # reused by CELL 5 (HQ-SAM)
DINO_CURT_BOXES = {}                                     # curtains — subtracted, reused too

RESULTS["A_grounded_sam2"] = {}
print("=== A) Grounded SAM 2  (grounding-dino-base + sam2.1_l) ===")
for it in IMAGES:
    t0 = time.time()
    boxes = dino_boxes(gd_proc, gd_model, it)            # see-through gated (no doors)
    cboxes = dino_boxes(gd_proc, gd_model, it, phrases=("curtain", "drape"), gate=0.0)
    DINO_BOXES[it["name"]] = boxes
    DINO_CURT_BOXES[it["name"]] = cboxes
    # window masks MINUS curtain masks — a DINO window box includes the sheer
    # curtains hanging over it; without this subtraction they end up as "glass"
    raw = sam2_union(sam2, it["det"], boxes) & ~sam2_union(sam2, it["det"], cboxes)
    glass = refine_glass(raw, it)
    RESULTS["A_grounded_sam2"][it["name"]] = glass
    print(f"  {len(boxes)} window + {len(cboxes)} curtain boxes, {time.time()-t0:.1f}s")
    show_pipeline("A GroundedSAM2", (220, 220, 0), it, raw, glass)

if not KEEP_MODELS:
    del sam2, gd_model, gd_proc                          # boxes live on in DINO_BOXES
    free_gpu()


# ============================== CELL 5 — B: DINO + HQ-SAM ViT-H ===============
# Same DINO boxes as CELL 4 (reused), masks from HQ-SAM (NeurIPS 2023) — trained
# to SHARPEN boundaries. checkpoint: lkeab/hq-sam sam_hq_vit_h.pth  [Apache-2.0]
from huggingface_hub import hf_hub_download
from segment_anything_hq import sam_model_registry, SamPredictor

if "DINO_BOXES" not in globals() or not DINO_BOXES:      # CELL 4 skipped? detect here
    gd_proc, gd_model = load_dino()
    DINO_BOXES = {it["name"]: dino_boxes(gd_proc, gd_model, it) for it in IMAGES}
    DINO_CURT_BOXES = {it["name"]: dino_boxes(gd_proc, gd_model, it,
                                              phrases=("curtain", "drape"), gate=0.0)
                       for it in IMAGES}
    if not KEEP_MODELS:
        del gd_model, gd_proc
        free_gpu()

ckpt = hf_hub_download("lkeab/hq-sam", "sam_hq_vit_h.pth")
samhq = sam_model_registry["vit_h"](checkpoint=ckpt).to(DEVICE).eval()
predictor = SamPredictor(samhq)

def hq_union(rgb, boxes):
    """HQ-SAM: union of masks for the given boxes (image must be set already)."""
    if not len(boxes):
        return np.zeros(rgb.shape[:2], bool)
    tb = predictor.transform.apply_boxes_torch(
        torch.as_tensor(boxes, dtype=torch.float, device=DEVICE), rgb.shape[:2])
    with torch.no_grad():
        masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None,
                                              boxes=tb, multimask_output=False,
                                              hq_token_only=True)
    return masks.cpu().numpy().astype(bool).any(axis=(0, 1))

RESULTS["B_hq_sam"] = {}
print("=== B) Grounding DINO + HQ-SAM ViT-H  (lkeab/hq-sam) ===")
for it in IMAGES:
    t0 = time.time()
    boxes = DINO_BOXES.get(it["name"], np.zeros((0, 4)))
    cboxes = DINO_CURT_BOXES.get(it["name"], np.zeros((0, 4)))
    rgb = cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    raw = hq_union(rgb, boxes) & ~hq_union(rgb, cboxes)  # windows minus curtains
    glass = refine_glass(raw, it)
    RESULTS["B_hq_sam"][it["name"]] = glass
    print(f"  {len(boxes)} boxes, {time.time()-t0:.1f}s")
    show_pipeline("B HQ-SAM", (0, 200, 255), it, raw, glass)

if not KEEP_MODELS:
    del predictor, samhq
    free_gpu()


# ============================== CELL 6 — C: SAM 3 (text -> masks) =============
# Meta SAM 3 (Nov 2025): Promptable Concept Segmentation — text prompt straight
# in, EVERY matching instance comes back. One model, no DINO.  [SAM license]
#
# facebook/sam3 is a GATED repo (free, but Meta's license must be accepted once):
#   1) log in at huggingface.co and open https://huggingface.co/facebook/sam3
#      -> click "Agree and access repository" (check huggingface.co/settings/
#      gated-repos until it says "Accepted")
#   2) make a READ token at https://huggingface.co/settings/tokens
#   3) in Colab: left sidebar key icon (Secrets) -> add HF_TOKEN = your token,
#      "Notebook access" ON — or run this cell and paste the token when asked.
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

RESULTS["C_sam3"] = {}
print("=== C) SAM 3  (facebook/sam3, text-prompted) ===")
for it in IMAGES:
    t0 = time.time()
    pil = Image.fromarray(cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB))
    raw = np.zeros(it["dark"].shape[:2], bool)
    for phrase in ("window pane", "window", "glass door"):
        inp = s3_proc(images=pil, text=phrase, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = s3_model(**inp)
        # threshold 0.35 (was 0.5): on the lifted dark bracket SAM 3 was badly
        # under-detecting (0.3% on the 3T1A3833 test). Each instance must still
        # GLOW in the dark image (see-through gate) so doors/walls stay out.
        res = s3_proc.post_process_instance_segmentation(
            out, threshold=0.35, mask_threshold=0.5,
            target_sizes=[pil.size[::-1]])[0]
        m = res.get("masks", None)
        if m is not None and len(m):
            arr = (m if hasattr(m, "cpu") else torch.stack(list(m))).cpu().numpy()
            for mi in arr.reshape(-1, *arr.shape[-2:]).astype(bool):
                if mi.any() and float(it["see"][mi].mean()) >= 0.06:
                    raw |= mi
    glass = refine_glass(raw, it)
    RESULTS["C_sam3"][it["name"]] = glass
    print(f"  {time.time()-t0:.1f}s")
    show_pipeline("C SAM3", (255, 150, 0), it, raw, glass)

if not KEEP_MODELS:
    del s3_model, s3_proc
    free_gpu()


# ============================== CELL 7 — D: Florence-2 + SAM 2.1 ==============
# Florence-2-large (MIT!): open-vocabulary detection by TEXT GENERATION — different
# failure modes than DINO, best commercial license. Boxes -> SAM 2.1.
# NOTE: uses the NATIVE transformers port (florence-community/Florence-2-large).
# The old microsoft/Florence-2-large remote code CRASHES on transformers v5
# ('Florence2LanguageConfig' has no attribute 'forced_bos_token_id') — same
# weights, working code, still MIT.
from transformers import AutoProcessor, Florence2ForConditionalGeneration
from ultralytics import SAM as _SAM

FL_ID = "florence-community/Florence-2-large"
fl_proc = AutoProcessor.from_pretrained(FL_ID)
fl_model = Florence2ForConditionalGeneration.from_pretrained(FL_ID).to(DEVICE).eval()
sam2_d = sam2 if "sam2" in globals() else _SAM("sam2.1_l.pt")

def florence_boxes(pil, phrase):
    task = "<OPEN_VOCABULARY_DETECTION>"
    inp = fl_proc(text=task + phrase, images=pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        gen = fl_model.generate(**inp, max_new_tokens=1024, do_sample=False, num_beams=3)
    txt = fl_proc.batch_decode(gen, skip_special_tokens=False)[0]
    parsed = fl_proc.post_process_generation(txt, task=task, image_size=(pil.width, pil.height))
    return np.array(parsed.get(task, {}).get("bboxes", []), np.float32).reshape(-1, 4)

RESULTS["D_florence2_sam2"] = {}
print("=== D) Florence-2-large + SAM 2.1  (MIT) ===")
for it in IMAGES:
    t0 = time.time()
    pil = Image.fromarray(cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB))
    allb = [florence_boxes(pil, p) for p in ("window pane", "window", "glass door")]
    boxes = np.vstack([b for b in allb if len(b)]) if any(len(b) for b in allb) else np.zeros((0, 4))
    if len(boxes):                                       # same see-through gate as DINO:
        keep = np.array([see_frac(b, it) >= 0.06 for b in boxes])
        if (~keep).sum():
            print(f"    {int((~keep).sum())} dark box(es) rejected (no light inside)")
        boxes = boxes[keep]
    cboxes = florence_boxes(pil, "curtain")              # curtains subtracted, like A/B
    raw = sam2_union(sam2_d, it["det"], boxes) & ~sam2_union(sam2_d, it["det"], cboxes)
    glass = refine_glass(raw, it)
    RESULTS["D_florence2_sam2"][it["name"]] = glass
    print(f"  {len(boxes)} boxes, {time.time()-t0:.1f}s")
    show_pipeline("D Florence2+SAM", (0, 90, 255), it, raw, glass)

if not KEEP_MODELS:
    del fl_model, fl_proc, sam2_d
    free_gpu()


# ============================== CELL 8 — E: OneFormer (ADE20K) ================
# Strongest ADE20K semantic model with a native "windowpane" class (58.3 mIoU).
# Tries the DiNAT-L backbone (needs natten); falls back to Swin-L automatically.
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

RESULTS["E_oneformer"] = {}
print("=== E) OneFormer ADE20K — class 8 'windowpane' ===")
for it in IMAGES:
    t0 = time.time()
    pil = Image.fromarray(cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB))
    inp = of_proc(images=pil, task_inputs=["semantic"], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = of_model(**inp)
    seg = of_proc.post_process_semantic_segmentation(out, target_sizes=[pil.size[::-1]])[0]
    raw = ade_windowpane(seg.cpu().numpy(), it)
    glass = refine_glass(raw, it)
    RESULTS["E_oneformer"][it["name"]] = glass
    print(f"  {time.time()-t0:.1f}s")
    show_pipeline("E OneFormer", (200, 0, 200), it, raw, glass)

if not KEEP_MODELS:
    del of_model, of_proc
    free_gpu()


# ============================== CELL 9 — F: Mask2Former (ADE20K) ==============
# facebook/mask2former-swin-large-ade-semantic (56.1 mIoU, "windowpane" native).
# !! weights are CC BY-NC (NON-commercial) — testing only, never ship this one.
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

M2F_ID = "facebook/mask2former-swin-large-ade-semantic"
m2f_proc = AutoImageProcessor.from_pretrained(M2F_ID)
m2f_model = Mask2FormerForUniversalSegmentation.from_pretrained(M2F_ID).to(DEVICE).eval()

RESULTS["F_mask2former"] = {}
print("=== F) Mask2Former ADE20K — class 8 'windowpane' ===")
for it in IMAGES:
    t0 = time.time()
    pil = Image.fromarray(cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB))
    inp = m2f_proc(images=pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = m2f_model(**inp)
    seg = m2f_proc.post_process_semantic_segmentation(out, target_sizes=[pil.size[::-1]])[0]
    raw = ade_windowpane(seg.cpu().numpy(), it)
    glass = refine_glass(raw, it)
    RESULTS["F_mask2former"][it["name"]] = glass
    print(f"  {time.time()-t0:.1f}s")
    show_pipeline("F Mask2Former", (0, 200, 0), it, raw, glass)

if not KEEP_MODELS:
    del m2f_model, m2f_proc
    free_gpu()


# ============================== CELL 10 — G: EoMT (ADE20K, CVPR 2025) =========
# tue-mps/ade20k_semantic_eomt_large_512 — "Your ViT is Secretly an Image
# Segmentation Model": a plain DINOv2 ViT-L segments by itself, ~4x faster than
# decoder models at similar quality. Native "windowpane" class.       [MIT]
from transformers import AutoImageProcessor as _AIP, EomtForUniversalSegmentation

EOMT_ID = "tue-mps/ade20k_semantic_eomt_large_512"
eo_proc = _AIP.from_pretrained(EOMT_ID)
eo_model = EomtForUniversalSegmentation.from_pretrained(EOMT_ID).to(DEVICE).eval()

RESULTS["G_eomt"] = {}
print("=== G) EoMT ADE20K — class 8 'windowpane' ===")
for it in IMAGES:
    t0 = time.time()
    pil = Image.fromarray(cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB))
    inp = eo_proc(images=pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = eo_model(**inp)
    seg = eo_proc.post_process_semantic_segmentation(out, target_sizes=[(pil.height, pil.width)])[0]
    raw = ade_windowpane(seg.cpu().numpy(), it)
    glass = refine_glass(raw, it)
    RESULTS["G_eomt"][it["name"]] = glass
    print(f"  {time.time()-t0:.1f}s")
    show_pipeline("G EoMT", (180, 120, 255), it, raw, glass)

if not KEEP_MODELS:
    del eo_model, eo_proc
    free_gpu()


# ============================== CELL 11 — H: SegGPT (in-context) ==============
# BAAI/seggpt-vit-large — segments BY EXAMPLE: upload ONE example pair —
#   1st file: an example DARKER image (any past scene)
#   2nd file: its glass mask (WHITE = glass, BLACK = rest — your marked PNG)
# and SegGPT copies that segmentation onto every new image. No text, no classes —
# it learns "what you mean" from your own marking.                    [MIT]
# (Press Cancel on the upload dialog to SKIP this pipeline.)
from transformers import SegGptImageProcessor, SegGptForImageSegmentation

print(">>> upload the EXAMPLE pair now: 1) example darker image  2) its white glass mask")
exup = files.upload()
if len(exup) < 2:
    print("SegGPT SKIPPED (need 2 files: example image + example mask)")
else:
    ex_paths = sorted(exup.keys())
    # the mask is the file that is (near-)binary; the other file is the image
    def _is_masky(p):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        return g is not None and float(((g < 40) | (g > 215)).mean()) > 0.9
    mask_path = next((p for p in ex_paths if _is_masky(p)), ex_paths[1])
    img_path = next(p for p in ex_paths if p != mask_path)

    ex_im = cv2.imread(img_path)
    hh, ww = ex_im.shape[:2]
    if ww != W:
        ex_im = cv2.resize(ex_im, (W, int(hh * W / ww)), interpolation=cv2.INTER_AREA)
    ex_det = (np.clip((ex_im.astype(np.float32) / 255) ** 0.5, 0, 1) * 255).astype("uint8")
    ex_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ex_mask = cv2.resize(ex_mask, (ex_im.shape[1], ex_im.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    ex_pil = Image.fromarray(cv2.cvtColor(ex_det, cv2.COLOR_BGR2RGB))
    ex_mask_pil = Image.fromarray(((ex_mask > 127) * 255).astype("uint8"), mode="L")
    print(f"example: image '{img_path}', mask '{mask_path}' "
          f"({(ex_mask > 127).mean()*100:.1f}% glass)")

    sg_proc = SegGptImageProcessor.from_pretrained("BAAI/seggpt-vit-large")
    sg_model = SegGptForImageSegmentation.from_pretrained("BAAI/seggpt-vit-large").to(DEVICE).eval()

    RESULTS["H_seggpt"] = {}
    print("=== H) SegGPT — in-context from YOUR example ===")
    for it in IMAGES:
        t0 = time.time()
        pil = Image.fromarray(cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB))
        inp = sg_proc(images=pil, prompt_images=ex_pil, prompt_masks=ex_mask_pil,
                      return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = sg_model(**inp)
        seg = sg_proc.post_process_semantic_segmentation(
            out, target_sizes=[(pil.height, pil.width)])[0]
        raw = seg.cpu().numpy() > 0
        glass = refine_glass(raw, it)
        RESULTS["H_seggpt"][it["name"]] = glass
        print(f"  {time.time()-t0:.1f}s")
        show_pipeline("H SegGPT", (255, 0, 150), it, raw, glass)

    if not KEEP_MODELS:
        del sg_model, sg_proc
        free_gpu()


# ============================== CELL 12 — comparison + agreement + zip ========
# For every image: all overlays in a grid + an AGREEMENT map —
#   GREEN = majority of the pipelines agree (almost certainly glass)
#   YELLOW = some agree (2 .. majority-1)      RED = only 1 says glass (suspect)
ORDER = ["A_grounded_sam2", "B_hq_sam", "C_sam3", "D_florence2_sam2",
         "E_oneformer", "F_mask2former", "G_eomt", "H_seggpt"]
PIPES = [k for k in ORDER if k in RESULTS and RESULTS[k]]
NEED = max(2, len(PIPES) // 2 + 1)                       # majority vote threshold
print(f"pipelines compared: {PIPES}  (consensus needs >= {NEED} votes)")

for it in IMAGES:
    name = it["name"]
    tiles = []
    for k in PIPES:
        slug = k.split("_")[0].lower()                   # "A_grounded_sam2" -> "a"
        t = cv2.imread(f"out_all/{name}_{slug}_overlay.jpg")
        if t is not None:
            tiles.append(cv2.resize(t, (1000, int(t.shape[0] * 1000 / t.shape[1]))))
    if tiles:
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
        if len(rows) > 1 and rows[-1].shape[1] != rows[0].shape[1]:
            pad = np.zeros((rows[-1].shape[0], rows[0].shape[1] - rows[-1].shape[1], 3), np.uint8)
            rows[-1] = np.hstack([rows[-1], pad])
        cv2.imwrite(f"out_all/{name}_GRID.jpg", np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    votes = np.zeros(it["dark"].shape[:2], np.uint8)
    for k in PIPES:
        votes += RESULTS[k][name].astype(np.uint8)
    agree = it["det"].astype(np.float32)
    for msk, col in (((votes == 1), (0, 0, 255)),
                     ((votes >= 2) & (votes < NEED), (0, 220, 220)),
                     ((votes >= NEED), (0, 200, 0))):
        a = msk.astype(np.float32)[..., None] * 0.55
        t = np.zeros_like(agree); t[:] = col
        agree = agree * (1 - a) + t * a
    agree = agree.clip(0, 255).astype("uint8")
    cv2.rectangle(agree, (0, 0), (agree.shape[1] - 1, 44), (0, 0, 0), -1)
    cv2.putText(agree, f"AGREEMENT {name}: green >={NEED} | yellow 2-{NEED-1} | red 1",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(f"out_all/{name}_AGREEMENT.jpg", agree, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # consensus mask (majority vote) — the safest glass mask for the window pull
    cv2.imwrite(f"out_all/{name}_CONSENSUS_mask.png",
                ((votes >= NEED).astype(np.uint8) * 255))

    print(f"\n{name}: glass area per pipeline — " +
          ", ".join(f"{k.split('_')[0]}={RESULTS[k][name].mean()*100:.1f}%" for k in PIPES) +
          f", consensus(>={NEED})={(votes>=NEED).mean()*100:.1f}%")
    try:
        from IPython.display import Image as _Img, display
        display(_Img(f"out_all/{name}_GRID.jpg"))
        display(_Img(f"out_all/{name}_AGREEMENT.jpg"))
    except Exception:
        pass

np.savez_compressed("out_all/glass_masks.npz",
                    **{f"{k}__{n}": (m.astype(np.uint8) * 255)
                       for k, d in RESULTS.items() for n, m in d.items()})
import shutil
shutil.make_archive("glass_all_models", "zip", "out_all")
files.download("glass_all_models.zip")
print("\ndone — glass_all_models.zip: per-model overlays + masks, grids, agreement "
      "maps, consensus masks and the .npz with every mask")
