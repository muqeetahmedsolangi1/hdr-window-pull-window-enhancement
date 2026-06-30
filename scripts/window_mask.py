"""Semantic masks via SegFormer (ADE20K-pretrained).

Provides soft masks for windows (white-balance exclusion), walls+ceiling
(forced white-balance) and floors (protected from neutralization), plus
lamp/cabinet/notwindow helpers.

The model downloads automatically from Hugging Face on first use and runs
on CPU. If torch/transformers are not installed, the getters return None
and the pipeline falls back to classical luminosity logic.

(Verbatim from the pixelswift project so hdr-enhance reproduces the
first-checkbox "HDR enhance" result exactly.)
"""
import os

import cv2
import numpy as np

# ADE20K class ids
WINDOW_CLASSES = (8, 14)        # windowpane, door (glass doors)
WALLCEIL_CLASSES = (0, 5)       # wall, ceiling
FLOOR_CLASSES = (3, 28)         # floor, rug
LAMP_CLASSES = (36, 82, 85)     # lamp, light/sconce, chandelier
CABINET_CLASSES = (10, 15, 24)  # cabinet, table, shelf (warm wood surfaces)
# things that are NOT windows but often reflect light / get mislabelled as a
# window: framed pictures/paintings, mirrors, glass tables, lamps. Excluded
# from the window mask so a sunlit picture frame or a glass table-top is never
# cropped/enhanced as a window.
NOTWINDOW_CLASSES = (22, 27, 36, 82, 85, 15)  # painting, mirror, lamps, table

# Stronger model: SegFormer-b5 (same ADE20K classes/API as b0, but far more
# accurate) correctly labels framed pictures as 'painting' and mirrors as
# 'mirror' instead of mislabelling them 'windowpane' — so they are excluded
# from window detection. Slower on CPU (~5-10s) and a bigger one-time download.
# Falls back to the small fast b0 if the big model can't be loaded. Override
# with AUTOHDR_SEG_MODEL (e.g. set it back to the b0 id for speed).
MODEL_ID = os.environ.get(
    "AUTOHDR_SEG_MODEL", "nvidia/segformer-b5-finetuned-ade-640-640")
FALLBACK_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
        for mid in (MODEL_ID, FALLBACK_ID):
            try:
                _processor = SegformerImageProcessor.from_pretrained(mid)
                _model = SegformerForSemanticSegmentation.from_pretrained(mid)
                _model.eval()
                print(f" * segmentation model: {mid}")
                break
            except Exception as e:
                print(f" * seg model {mid} failed ({e}); trying fallback")
                _model = None
        if _model is None:
            raise RuntimeError("no segmentation model could be loaded")
    return _model, _processor


def get_label_masks(bgr):
    """Dict of soft float32 masks {'window','wallceil','floor','lamp',
    'cabinet','notwindow'} at full resolution, or None if segmentation is
    unavailable. Individual entries are None when that class is absent."""
    try:
        import torch
        model, processor = _load()
    except Exception:
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits  # 1 x classes x h/4 x w/4
    labels = logits.argmax(dim=1)[0].numpy().astype(np.uint8)

    h, w = bgr.shape[:2]

    def soft(classes):
        m = np.isin(labels, classes).astype(np.float32)
        if m.sum() < 10:
            return None
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(m, 0, 1)

    return {
        "window": soft(WINDOW_CLASSES),
        "wallceil": soft(WALLCEIL_CLASSES),
        "floor": soft(FLOOR_CLASSES),
        "lamp": soft(LAMP_CLASSES),
        "cabinet": soft(CABINET_CLASSES),
        "notwindow": soft(NOTWINDOW_CLASSES),
    }


def get_window_mask(bgr, feather=51):
    """Back-compat: soft window mask only."""
    masks = get_label_masks(bgr)
    return None if masks is None else masks["window"]
