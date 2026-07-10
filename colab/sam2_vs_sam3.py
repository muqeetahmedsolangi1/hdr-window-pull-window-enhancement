"""
SAM 2 vs SAM 3 — head-to-head WINDOW GLASS detection on the DARKER exposure.

Upload the darker bracket(s) (where the outdoor view is clear); each model finds
ONLY the see-through glass — the pane area where the outdoor view shows:

  CELL 4 — SAM 2 pipeline: Grounding DINO (text -> boxes) + SAM 2.1 Hiera-Large.
           SAM 2 cannot read text, so DINO finds the windows first.        [Apache-2.0]
  CELL 5 — SAM 3 pipeline: ONE model, text prompt straight in ("window pane"),
           every matching instance comes back as a mask.                  [SAM license]

Both get the SAME treatment so the fight is fair:
  * models LOOK at a gamma-lifted copy (a dark bracket hides windows from any
    model); all masks apply to the ORIGINAL dark image
  * the same glass-refine: thin colourless frame/mullion bars removed, thick
    shaded view patches kept, specks dropped
  * works on one darker image or many at once (it loops)

CELL 6 draws the DUEL map — green where both agree, orange = only SAM 2,
blue = only SAM 3 — and zips every mask/overlay for download.

Run on Google Colab with a GPU (A100 or T4). Copy each CELL into its own cell.
"""

# ============================== CELL 1 — install ==============================
# (Runtime -> Change runtime type -> GPU, first!)
# !pip install -q -U transformers ultralytics opencv-python-headless
# !pip install -q --force-reinstall "pillow==10.4.0"
#
# >>> AFTER THIS CELL: Runtime -> Restart session, then run CELL 2. <<<


# ============================== CELL 2 — upload the DARKER image(s) ===========
import os, time, torch, numpy as np, cv2
from PIL import Image
from google.colab import files

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| gpu:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "-")

up = files.upload()
if not up:
    raise RuntimeError("no file uploaded — run this cell again and pick the darker image(s)")

W = 2000
IMAGES = []                       # [{name, dark(BGR), det(BGR gamma-lifted)}]
for _name in sorted(up.keys()):
    im = cv2.imread(_name)
    if im is None:
        print(f"  ! could not read '{_name}' — skipped"); continue
    h, w = im.shape[:2]
    if w != W:
        im = cv2.resize(im, (W, int(h * W / w)), interpolation=cv2.INTER_AREA)
    # models must SEE: a dark bracket hides the dimmer windows from ANY model —
    # both look at this gamma-lifted copy; masks/physics use the ORIGINAL.
    det = (np.clip((im.astype(np.float32) / 255) ** 0.5, 0, 1) * 255).astype("uint8")
    IMAGES.append({"name": os.path.splitext(_name)[0], "dark": im, "det": det})
    print(f"  {_name}: {im.shape[1]}x{im.shape[0]} (mean {im.mean():.0f}/255)")

RESULTS = {}                      # RESULTS[model][image_name] = bool glass mask
os.makedirs("out_duel", exist_ok=True)
try:
    from IPython.display import Image as _Img, display
    for it in IMAGES:
        cv2.imwrite("_show.jpg", it["dark"], [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"darker input '{it['name']}' (THIS gets segmented):")
        display(_Img("_show.jpg"))
except Exception:
    pass


# ============================== CELL 3 — shared helpers =======================
def refine_glass(raw, it):
    """Same cleanup for BOTH models (fair fight), on the ORIGINAL dark image:
    remove thin DARK colourless frame/mullion bars; KEEP anything lit (venetian
    blind slats!) and thick colourless patches (shaded view); drop specks.
    Bars must be DARK — an any-brightness rule eats entire venetian blinds."""
    H, Wd = raw.shape
    d = it["dark"].astype(np.float32) / 255
    dl = d.mean(2); dc = d.max(2) - d.min(2)
    m = raw.astype(bool).copy()
    achro = (dc < 0.07) & (dl < 0.18) & m
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


def show_model(tag, colour, it, raw, glass):
    """Overlay on the lifted copy (visible) — faint tint = dropped by the refine,
    solid tint + contour = final glass. Saves overlay JPG + binary mask PNG."""
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
    slug = tag.lower().replace(" ", "")
    cv2.imwrite(f"out_duel/{it['name']}_{slug}_overlay.jpg", o, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(f"out_duel/{it['name']}_{slug}_mask.png", glass.astype(np.uint8) * 255)
    print(f"  {tag:6s} {it['name']}: raw {raw.mean()*100:5.1f}% -> glass {glass.mean()*100:5.1f}%")
    try:
        from IPython.display import Image as _Img, display
        display(_Img(f"out_duel/{it['name']}_{slug}_overlay.jpg"))
    except Exception:
        pass

print("helpers ready")


# ============================== CELL 4 — SAM 2 (DINO + SAM 2.1-L) =============
# SAM 2 cannot take a text prompt, so Grounding DINO finds the window boxes and
# SAM 2.1 Hiera-Large turns each box into a pixel-precise mask.
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import SAM

GD_ID = "IDEA-Research/grounding-dino-base"
gd_proc = AutoProcessor.from_pretrained(GD_ID)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_ID).to(DEVICE).eval()
sam2 = SAM("sam2.1_l.pt")                                # auto-downloads the checkpoint

RESULTS["SAM2"] = {}
print("=== SAM 2 pipeline (Grounding DINO + SAM 2.1-L) ===")
for it in IMAGES:
    t0 = time.time()
    pil = Image.fromarray(cv2.cvtColor(it["det"], cv2.COLOR_BGR2RGB))
    # synonym-ensemble detection: union the boxes from every phrasing
    allb = []
    for phrase in ("window", "windowpane", "glass door"):
        inp = gd_proc(images=pil, text=phrase + " .", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = gd_model(**inp)
        try:
            det = gd_proc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=0.25, text_threshold=0.20,
                target_sizes=[pil.size[::-1]])[0]
        except TypeError:
            det = gd_proc.post_process_grounded_object_detection(
                out, inp.input_ids, box_threshold=0.25, text_threshold=0.20,
                target_sizes=[pil.size[::-1]])[0]
        if len(det["boxes"]):
            allb.append(det["boxes"].cpu().numpy())
    boxes = np.vstack(allb) if allb else np.zeros((0, 4))

    raw = np.zeros(it["dark"].shape[:2], bool)
    if len(boxes):
        try:
            r = sam2(it["det"], bboxes=boxes.tolist(), retina_masks=True, verbose=False)
        except TypeError:
            r = sam2(it["det"], bboxes=boxes.tolist(), verbose=False)
        if r and r[0].masks is not None:
            raw = r[0].masks.data.cpu().numpy().astype(bool).any(axis=0)
    glass = refine_glass(raw, it)
    RESULTS["SAM2"][it["name"]] = glass
    print(f"  {len(boxes)} boxes, {time.time()-t0:.1f}s")
    show_model("SAM2", (220, 220, 0), it, raw, glass)


# ============================== CELL 5 — SAM 3 (text -> masks) ================
# Meta's SAM 3 (Nov 2025): Promptable Concept Segmentation — the text prompt goes
# straight in, every matching instance comes back as a mask. No DINO needed.
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

RESULTS["SAM3"] = {}
print("=== SAM 3 pipeline (text-prompted, single model) ===")
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
    RESULTS["SAM3"][it["name"]] = glass
    print(f"  {time.time()-t0:.1f}s")
    show_model("SAM3", (255, 150, 0), it, raw, glass)


# ============================== CELL 6 — DUEL map + zip + download ============
# GREEN = both agree (safe glass) · ORANGE = only SAM 2 · BLUE = only SAM 3
for it in IMAGES:
    name = it["name"]
    m2 = RESULTS["SAM2"].get(name, np.zeros(it["dark"].shape[:2], bool))
    m3 = RESULTS["SAM3"].get(name, np.zeros(it["dark"].shape[:2], bool))
    duel = it["det"].astype(np.float32)
    for msk, col in ((m2 & m3, (0, 200, 0)), (m2 & ~m3, (0, 165, 255)), (m3 & ~m2, (255, 150, 0))):
        a = msk.astype(np.float32)[..., None] * 0.55
        t = np.zeros_like(duel); t[:] = col
        duel = duel * (1 - a) + t * a
    duel = duel.clip(0, 255).astype("uint8")
    cv2.rectangle(duel, (0, 0), (duel.shape[1] - 1, 44), (0, 0, 0), -1)
    cv2.putText(duel, f"DUEL {name}: green both | orange SAM2 only | blue SAM3 only",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.imwrite(f"out_duel/{name}_DUEL.jpg", duel, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(f"out_duel/{name}_BOTH_mask.png", ((m2 & m3).astype(np.uint8) * 255))

    inter = float((m2 & m3).sum()); union = float((m2 | m3).sum())
    iou = inter / union if union else 0.0
    print(f"\n{name}: SAM2 {m2.mean()*100:.1f}% | SAM3 {m3.mean()*100:.1f}% | "
          f"agree {inter/max(m2.shape[0]*m2.shape[1],1)*100:.1f}% | IoU {iou:.2f}")
    try:
        from IPython.display import Image as _Img, display
        display(_Img(f"out_duel/{name}_DUEL.jpg"))
    except Exception:
        pass

np.savez_compressed("out_duel/duel_masks.npz",
                    **{f"{k}__{n}": (m.astype(np.uint8) * 255)
                       for k, d in RESULTS.items() for n, m in d.items()})
import shutil
shutil.make_archive("sam2_vs_sam3", "zip", "out_duel")
files.download("sam2_vs_sam3.zip")
print("\ndone — sam2_vs_sam3.zip: overlays + masks for both models, DUEL maps, "
      "BOTH-agree masks and the .npz")
