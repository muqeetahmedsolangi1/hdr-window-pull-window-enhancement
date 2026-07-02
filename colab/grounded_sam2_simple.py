"""
Grounded-SAM 2 — SIMPLE version (single image -> one combined mask overlay).
Quick test of "type a concept, get a precise mask". For the DETAILED per-concept +
fused-image version, use grounded_sam2_masks.py instead.

Run on Google Colab with a GPU (A100 / T4). Copy each CELL into its own Colab cell.
Pipeline:  Grounding DINO (text -> boxes)  ->  SAM 2.1 (boxes -> pixel-perfect masks)
"""

# ============================== CELL 1 — install ==============================
# (Runtime -> Change runtime type -> GPU : A100 or T4, first!)
# !pip install -q -U transformers ultralytics supervision opencv-python-headless
# !pip install -q --force-reinstall "pillow==10.4.0"
#
# >>> AFTER THIS CELL: Runtime -> Restart session  (MUST restart, then run CELL 2). <<<


# ============================== CELL 2 — load models ==========================
import torch, numpy as np, cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import SAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| gpu:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "-")

GD_ID = "IDEA-Research/grounding-dino-base"
gd_proc = AutoProcessor.from_pretrained(GD_ID)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_ID).to(DEVICE).eval()
sam = SAM("sam2.1_l.pt")                             # SAM 2.1 Hiera-Large (auto-downloads)
print("models loaded")


# ============================== CELL 3 — upload a single image ================
from google.colab import files
up = files.upload()
IMG_PATH = list(up.keys())[0]
pil = Image.open(IMG_PATH).convert("RGB")
bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
print("image:", pil.size)


# ============================== CELL 4 — prompt + segment =====================
PROMPT = "curtain. blind. window. window frame. sky. water. tree. door."
BOX_THRESH, TEXT_THRESH = 0.30, 0.25

inputs = gd_proc(images=pil, text=PROMPT, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = gd_model(**inputs)
try:
    det = gd_proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=BOX_THRESH,
        text_threshold=TEXT_THRESH, target_sizes=[pil.size[::-1]])[0]
except TypeError:
    det = gd_proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=BOX_THRESH,
        text_threshold=TEXT_THRESH, target_sizes=[pil.size[::-1]])[0]
boxes = det["boxes"].cpu().numpy()
labels = det.get("text_labels", det.get("labels"))
scores = det["scores"].cpu().numpy()
print(f"found {len(boxes)} objects:")
for l, s in zip(labels, scores):
    print(f"  {l:20s} {s:.2f}")

masks = np.zeros((0, *bgr.shape[:2]), bool)
if len(boxes):
    sres = sam(bgr, bboxes=boxes.tolist(), verbose=False)
    masks = sres[0].masks.data.cpu().numpy().astype(bool)


# ============================== CELL 5 — group by concept =====================
def norm(lbl):
    lbl = lbl.lower()
    for key in ("curtain", "blind", "window frame", "window", "sky", "water",
                "tree", "door"):
        if key in lbl:
            return key
    return lbl

concept = {}
for m, l in zip(masks, labels):
    k = norm(l)
    concept[k] = m if k not in concept else (concept[k] | m)
print("concepts:", {k: f"{v.mean()*100:.1f}%" for k, v in concept.items()})


# ============================== CELL 6 — visualise + export ===================
COLORS = {"curtain": (200, 0, 200), "blind": (200, 0, 120), "window": (200, 200, 0),
          "window frame": (0, 140, 255), "sky": (255, 150, 0), "water": (255, 255, 0),
          "tree": (0, 200, 0), "door": (0, 0, 220)}
ov = bgr.astype(np.float32)
for k, m in concept.items():
    col = np.array(COLORS.get(k, (255, 255, 255)), np.float32)
    a = m[..., None] * 0.5
    ov = ov * (1 - a) + col * a
ov = ov.clip(0, 255).astype(np.uint8)
cv2.imwrite("gsam2_overlay.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 92])

np.savez_compressed(
    "gsam2_masks.npz",
    **{k.replace(" ", "_"): (m.astype(np.uint8) * 255) for k, m in concept.items()})

try:
    from IPython.display import Image as _Img, display
    display(_Img("gsam2_overlay.jpg"))
except Exception:
    pass
files.download("gsam2_overlay.jpg")
files.download("gsam2_masks.npz")
print("done — overlay to eyeball, npz for the local pull")
