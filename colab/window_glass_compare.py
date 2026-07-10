"""
WINDOW GLASS DETECTION — 4-MODEL SHOOTOUT on the DARKER exposure.

You upload the DARKER bracket(s) (where the outdoor view is clear). Each pipeline
below detects ONLY the see-through glass — the pane area where the outdoor view
shows — never the frame, curtains or the wall:

  A) Grounded SAM 2 . Grounding DINO (text->boxes) + SAM 2.1 Hiera-Large   [Apache-2.0]
  B) DINO + HQ-SAM .. same boxes + HQ-SAM ViT-H (sharper mask boundaries)  [Apache-2.0]
  C) OneFormer ...... ADE20K semantic, class 8 "windowpane"  (58.3 mIoU)   [weights: research]
  D) Mask2Former .... ADE20K semantic, class 8 "windowpane"  (56.1 mIoU)   [weights: CC BY-NC]
  E) SAM 3 .......... Meta Nov-2025: text prompt -> ALL instance masks in  [SAM license]
                      ONE model (replaces the DINO+SAM combo of A/B)

Picked by the July-2026 research: Grounding DINO 1.5/1.6/DINO-X are API-only (no
weights released) and GDNet is stuck on PyTorch 1.0/CUDA 10 — both excluded.
SAM 3 post-dates that research shortlist and was added as the modern challenger.

Every pipeline gets the SAME treatment so the comparison is fair:
  * models LOOK at a gamma-lifted copy (a dark bracket hides windows from any model);
    all masks apply to the ORIGINAL dark image
  * the same glass-refine step: thin colourless frame/mullion bars are removed,
    thick shaded view patches are kept, specks dropped
  * on a big GPU (A100) every model STAYS loaded (instant re-runs); on a small
    one (T4) each cell frees its model after running — decided automatically
  * works on one image or many (upload several darker images at once — it loops)

CELL 9 shows all five masks side by side + an agreement map, and zips everything.

Run on Google Colab with a GPU. Copy each CELL into its own Colab cell.
"""

# ============================== CELL 1 — install ==============================
# (Runtime -> Change runtime type -> GPU : A100 or T4, first!)
# !pip install -q -U transformers ultralytics opencv-python-headless huggingface_hub
# !pip install -q segment-anything-hq timm
# !pip install -q --force-reinstall "pillow==10.4.0"
# # natten is OPTIONAL (only for the strongest OneFormer backbone; CELL 6 falls back
# # to the Swin backbone automatically if this line fails — just let it fail).
# # KNOWN: shi-labs.com's SSL certificate is expired -> "CERTIFICATE_VERIFY_FAILED"
# # retry warnings here are HARMLESS, everything else installed fine. To force it:
# # add  --trusted-host shi-labs.com  to the line below.
# !pip install -q natten -f https://shi-labs.com/natten/wheels
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
# T4 (16 GB): free each model after its cell so all four pipelines fit.
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

RESULTS = {}                      # RESULTS[pipeline][image_name] = bool mask
os.makedirs("out_compare", exist_ok=True)
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
    remove thin DARK colourless frame/mullion bars; KEEP anything lit (venetian
    blind slats!) and thick colourless patches (shaded view); drop specks.
    Bars must be DARK — an any-brightness rule eats entire venetian blinds."""
    H, Wd = raw.shape
    d = it["dark"].astype(np.float32) / 255
    dl = d.mean(2); dc = d.max(2) - d.min(2)
    m = raw.astype(bool).copy()
    achro = (dc < 0.07) & (dl < 0.18) & m                # DARK colourless pixels only
    bar_k = max(5, int(Wd * 0.014) | 1)
    blobs = cv2.morphologyEx(achro.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((bar_k, bar_k), np.uint8)).astype(bool)
    m &= ~(achro & ~blobs)                               # thin bars out, thick blobs stay
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    ncc, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    for i in range(1, ncc):                              # a real pane is never a speck
        if st[i, cv2.CC_STAT_AREA] < 0.0005 * H * Wd:
            m[lab == i] = False
    return m


def show_pipeline(tag, colour, it, raw, glass):
    """Overlay (on the lifted copy, so you can see) + binary mask, saved and shown."""
    base = it["det"].astype(np.float32)
    for msk, a in ((raw & ~glass, 0.25), (glass, 0.55)):     # faint = dropped by refine
        al = msk.astype(np.float32)[..., None] * a
        t = np.zeros_like(base); t[:] = colour
        base = base * (1 - al) + t * al
    o = base.clip(0, 255).astype("uint8")
    cnts, _ = cv2.findContours(glass.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cnts, -1, colour, 2)
    cv2.rectangle(o, (0, 0), (o.shape[1] - 1, 44), (0, 0, 0), -1)
    cv2.putText(o, f"{tag} - {it['name']} - glass {glass.mean()*100:.1f}%", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    slug = tag.split()[0].lower()
    cv2.imwrite(f"out_compare/{it['name']}_{slug}_overlay.jpg", o, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(f"out_compare/{it['name']}_{slug}_mask.png", glass.astype(np.uint8) * 255)
    print(f"  {tag:24s} {it['name']}: raw {raw.mean()*100:5.1f}% -> glass {glass.mean()*100:5.1f}%")
    try:
        from IPython.display import Image as _Img, display
        display(_Img(f"out_compare/{it['name']}_{slug}_overlay.jpg"))
    except Exception:
        pass


def load_dino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    GD_ID = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(GD_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_ID).to(DEVICE).eval()
    return proc, model


def dino_boxes(proc, model, bgr, box_t=0.25, text_t=0.20):
    """Synonym-ensemble window detection — union of boxes over several phrasings."""
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    allb = []
    for phrase in ("window", "windowpane", "glass door"):
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
    return np.vstack(allb) if allb else np.zeros((0, 4))


def ade_windowpane(seg_map, it):
    """ADE20K semantics -> glass: 'windowpane'(8) everywhere; 'door'(14) only where
    light actually shows through (a glass door passes, a solid wood door is dark)."""
    return (seg_map == 8) | ((seg_map == 14) & it["see"])

print("helpers ready")


# ============================== CELL 4 — A: Grounded SAM 2 ====================
# Grounding DINO (text -> boxes) + SAM 2.1 Hiera-Large (boxes -> pixel masks).
from ultralytics import SAM

gd_proc, gd_model = load_dino()
sam2 = SAM("sam2.1_l.pt")                               # auto-downloads the checkpoint
DINO_BOXES = {}                                          # kept for CELL 5 (HQ-SAM reuse)

RESULTS["A_grounded_sam2"] = {}
print("=== A) Grounded SAM 2 (DINO + SAM 2.1-L) ===")
for it in IMAGES:
    t0 = time.time()
    boxes = dino_boxes(gd_proc, gd_model, it["det"])
    DINO_BOXES[it["name"]] = boxes
    raw = np.zeros(it["dark"].shape[:2], bool)
    if len(boxes):
        try:
            r = sam2(it["det"], bboxes=boxes.tolist(), retina_masks=True, verbose=False)
        except TypeError:
            r = sam2(it["det"], bboxes=boxes.tolist(), verbose=False)
        if r and r[0].masks is not None:
            raw = r[0].masks.data.cpu().numpy().astype(bool).any(axis=0)
    glass = refine_glass(raw, it)
    RESULTS["A_grounded_sam2"][it["name"]] = glass
    print(f"  {len(boxes)} boxes, {time.time()-t0:.1f}s")
    show_pipeline("A GroundedSAM2", (220, 220, 0), it, raw, glass)

if not KEEP_MODELS:
    del sam2, gd_model, gd_proc                          # boxes live on in DINO_BOXES
    free_gpu()


# ============================== CELL 5 — B: DINO + HQ-SAM ViT-H ===============
# Same DINO boxes as CELL 4 (reused), but the masks come from HQ-SAM — trained to
# sharpen boundaries (the edge-quality challenger from the research).
from huggingface_hub import hf_hub_download
from segment_anything_hq import sam_model_registry, SamPredictor

if "DINO_BOXES" not in globals() or not DINO_BOXES:      # CELL 4 skipped? detect here
    gd_proc, gd_model = load_dino()
    DINO_BOXES = {it["name"]: dino_boxes(gd_proc, gd_model, it["det"]) for it in IMAGES}
    if not KEEP_MODELS:
        del gd_model, gd_proc
        free_gpu()

ckpt = hf_hub_download("lkeab/hq-sam", "sam_hq_vit_h.pth")
samhq = sam_model_registry["vit_h"](checkpoint=ckpt).to(DEVICE).eval()
predictor = SamPredictor(samhq)

RESULTS["B_hq_sam"] = {}
print("=== B) Grounding DINO + HQ-SAM ViT-H ===")
for it in IMAGES:
    t0 = time.time()
    boxes = DINO_BOXES.get(it["name"], np.zeros((0, 4)))
    raw = np.zeros(it["dark"].shape[:2], bool)
    if len(boxes):
        rgb = cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        tb = predictor.transform.apply_boxes_torch(
            torch.as_tensor(boxes, dtype=torch.float, device=DEVICE), rgb.shape[:2])
        with torch.no_grad():
            masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None,
                                                  boxes=tb, multimask_output=False,
                                                  hq_token_only=True)
        raw = masks.cpu().numpy().astype(bool).any(axis=(0, 1))
    glass = refine_glass(raw, it)
    RESULTS["B_hq_sam"][it["name"]] = glass
    print(f"  {len(boxes)} boxes, {time.time()-t0:.1f}s")
    show_pipeline("B HQ-SAM", (0, 200, 255), it, raw, glass)

if not KEEP_MODELS:
    del predictor, samhq
    free_gpu()


# ============================== CELL 6 — C: OneFormer (ADE20K) ================
# Best ADE20K semantic model with a native "windowpane" class. Tries the strongest
# DiNAT-L backbone (needs natten); falls back to Swin-L automatically.
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

RESULTS["C_oneformer"] = {}
print("=== C) OneFormer ADE20K — class 8 'windowpane' ===")
for it in IMAGES:
    t0 = time.time()
    pil = Image.fromarray(cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB))
    inp = of_proc(images=pil, task_inputs=["semantic"], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = of_model(**inp)
    seg = of_proc.post_process_semantic_segmentation(out, target_sizes=[pil.size[::-1]])[0]
    raw = ade_windowpane(seg.cpu().numpy(), it)
    glass = refine_glass(raw, it)
    RESULTS["C_oneformer"][it["name"]] = glass
    print(f"  {time.time()-t0:.1f}s")
    show_pipeline("C OneFormer", (200, 0, 200), it, raw, glass)

if not KEEP_MODELS:
    del of_model, of_proc
    free_gpu()


# ============================== CELL 7 — D: Mask2Former (ADE20K) ==============
# NOTE: these pretrained weights are CC BY-NC (non-commercial) — testing only;
# don't ship this one in the commercial pipeline.
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

M2F_ID = "facebook/mask2former-swin-large-ade-semantic"
m2f_proc = AutoImageProcessor.from_pretrained(M2F_ID)
m2f_model = Mask2FormerForUniversalSegmentation.from_pretrained(M2F_ID).to(DEVICE).eval()

RESULTS["D_mask2former"] = {}
print("=== D) Mask2Former ADE20K — class 8 'windowpane' ===")
for it in IMAGES:
    t0 = time.time()
    pil = Image.fromarray(cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB))
    inp = m2f_proc(images=pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = m2f_model(**inp)
    seg = m2f_proc.post_process_semantic_segmentation(out, target_sizes=[pil.size[::-1]])[0]
    raw = ade_windowpane(seg.cpu().numpy(), it)
    glass = refine_glass(raw, it)
    RESULTS["D_mask2former"][it["name"]] = glass
    print(f"  {time.time()-t0:.1f}s")
    show_pipeline("D Mask2Former", (0, 200, 0), it, raw, glass)

if not KEEP_MODELS:
    del m2f_model, m2f_proc
    free_gpu()


# ============================== CELL 8 — E: SAM 3 (text -> masks) =============
# Meta's SAM 3 (Nov 2025): Promptable Concept Segmentation — the text prompt goes
# straight in and EVERY matching instance comes back as a mask. One model instead
# of the DINO+SAM pair. Needs a recent transformers (installed in CELL 1).
#
# facebook/sam3 is a GATED repo (free, but Meta's license must be accepted once):
#   1) log in at huggingface.co and open https://huggingface.co/facebook/sam3
#      -> click "Agree and access repository"
#   2) make a READ token at https://huggingface.co/settings/tokens
#   3) in Colab: left sidebar key icon (Secrets) -> add HF_TOKEN = your token,
#      switch "Notebook access" ON — or just run this cell and paste the token
#      into the login box it shows.
import os
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

RESULTS["E_sam3"] = {}
print("=== E) SAM 3 — text-prompted concept segmentation ===")
for it in IMAGES:
    t0 = time.time()
    pil = Image.fromarray(cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB))
    raw = np.zeros(it["dark"].shape[:2], bool)
    for phrase in ("window pane", "window", "glass door"):
        inp = s3_proc(images=pil, text=phrase, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = s3_model(**inp)
        res = s3_proc.post_process_instance_segmentation(
            out, threshold=0.5, mask_threshold=0.5,
            target_sizes=[pil.size[::-1]])[0]
        m = res.get("masks", None)
        if m is not None and len(m):
            arr = (m if hasattr(m, "cpu") else torch.stack(list(m))).cpu().numpy()
            raw |= arr.reshape(-1, *arr.shape[-2:]).astype(bool).any(axis=0)
    glass = refine_glass(raw, it)
    RESULTS["E_sam3"][it["name"]] = glass
    print(f"  {time.time()-t0:.1f}s")
    show_pipeline("E SAM3", (255, 150, 0), it, raw, glass)

if not KEEP_MODELS:
    del s3_model, s3_proc
    free_gpu()


# ============================== CELL 9 — side-by-side + agreement + zip =======
# For every image: the 5 overlays in a grid, plus an AGREEMENT map —
#   GREEN = 3+ models agree (almost certainly glass)
#   YELLOW = exactly 2 agree      RED = only 1 says glass (suspect)
PIPES = [k for k in ("A_grounded_sam2", "B_hq_sam", "C_oneformer", "D_mask2former",
                     "E_sam3")
         if k in RESULTS and RESULTS[k]]
print("pipelines compared:", PIPES)

for it in IMAGES:
    name = it["name"]
    tiles = []
    for k in PIPES:
        slug = {"A_grounded_sam2": "a", "B_hq_sam": "b", "C_oneformer": "c",
                "D_mask2former": "d", "E_sam3": "e"}[k]
        p = f"out_compare/{name}_{slug}_overlay.jpg"
        t = cv2.imread(p)
        if t is not None:
            tiles.append(cv2.resize(t, (1000, int(t.shape[0] * 1000 / t.shape[1]))))
    if tiles:
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
        if len(rows) > 1 and rows[-1].shape[1] != rows[0].shape[1]:
            pad = np.zeros((rows[-1].shape[0], rows[0].shape[1] - rows[-1].shape[1], 3), np.uint8)
            rows[-1] = np.hstack([rows[-1], pad])
        cv2.imwrite(f"out_compare/{name}_GRID.jpg", np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    votes = np.zeros(it["dark"].shape[:2], np.uint8)
    for k in PIPES:
        votes += RESULTS[k][name].astype(np.uint8)
    agree = it["det"].astype(np.float32)
    for lo, col in ((1, (0, 0, 255)), (2, (0, 220, 220)), (3, (0, 200, 0))):
        msk = (votes == lo) if lo < 3 else (votes >= 3)
        a = msk.astype(np.float32)[..., None] * 0.55
        t = np.zeros_like(agree); t[:] = col
        agree = agree * (1 - a) + t * a
    agree = agree.clip(0, 255).astype("uint8")
    cv2.rectangle(agree, (0, 0), (agree.shape[1] - 1, 44), (0, 0, 0), -1)
    cv2.putText(agree, f"AGREEMENT {name}: green 3+ | yellow 2 | red 1",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(f"out_compare/{name}_AGREEMENT.jpg", agree, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # consensus mask (>=3 votes) — the safest glass mask to feed the window pull
    cv2.imwrite(f"out_compare/{name}_CONSENSUS_mask.png",
                ((votes >= 3).astype(np.uint8) * 255))

    print(f"\n{name}: glass area per pipeline — " +
          ", ".join(f"{k.split('_')[0]}={RESULTS[k][name].mean()*100:.1f}%" for k in PIPES) +
          f", consensus(>=3)={(votes>=3).mean()*100:.1f}%")
    try:
        from IPython.display import Image as _Img, display
        display(_Img(f"out_compare/{name}_GRID.jpg"))
        display(_Img(f"out_compare/{name}_AGREEMENT.jpg"))
    except Exception:
        pass

np.savez_compressed("out_compare/glass_masks.npz",
                    **{f"{k}__{n}": (m.astype(np.uint8) * 255)
                       for k, d in RESULTS.items() for n, m in d.items()})
import shutil
shutil.make_archive("glass_compare", "zip", "out_compare")
files.download("glass_compare.zip")
print("\ndone — glass_compare.zip: per-model overlays + masks, grids, agreement maps, "
      "consensus masks and the .npz with every mask")
