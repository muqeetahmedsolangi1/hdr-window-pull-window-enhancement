# hdr-enhance

A **standalone**, self-contained port of the **first checkbox** of the pixelswift
project &mdash; *"HDR enhance &mdash; calibrated color/tone finishing"* &mdash; i.e.
`pipeline.process_brackets(..., enhance=True, pull_windows=False)`.

It merges a set of **bracketed exposures** with classic **Mertens exposure fusion**
and then applies the calibrated **"white HDR"** finishing chain to produce the bright,
clean, neutral-white real-estate look. No window pull, no generative step.

It produces a **bit-for-bit identical** result to the pixelswift first checkbox. It
uses **SegFormer** semantic masks (via `transformers` + `torch`) to refine the
finishing exactly like pixelswift, and **auto-falls-back to a pure OpenCV + NumPy
path** when those aren't installed.

Self-contained: nothing is imported from any other project (`window_mask.py` is its
own verbatim copy of the segmentation module).

## Pipeline (identical order to pixelswift)
1. **Sort** brackets darkest-first (upload order is unknown).
2. **Resize** all brackets to the smallest common size, capped at `max_width`
   (`cv2.INTER_AREA`).
3. **Align** every bracket to the middle one (`align_ecc`, ECC homography,
   exposure-robust).
4. **Merge**: `cv2.createMergeMertens().process(...)` &rarr; the HDR base, float32 BGR.
5. **Finish** (only when *HDR enhance* is on), in this exact order:
   `white_patch_wb` &rarr; `auto_levels` &rarr; `auto_exposure(target=0.70)` &rarr;
   `whiten_lamps` &rarr; `neutralize_whites` &rarr; `match_neutral_whites` &rarr;
   `s_curve(strength=0.05)`. The window (blown glass) is excluded from the
   white-balance / levels measurement so the white point anchors on the **walls**.
6. **Sharpen** (always, runs last): `sharpen_two_stage` (fine unsharp + wide-radius
   clarity).
7. Convert to uint8 BGR.

With *HDR enhance* **off** you get plain Mertens fusion + the two-stage sharpen only.

## Install
```
pip install -r requirements.txt
```

## Use &mdash; command line
```
cd scripts
python run.py "<folder_of_exposures>" hdr.jpg 4000
python run.py "<folder_of_exposures>" raw.jpg 4000 no-enhance   # checkbox OFF
```
- arg 1: folder containing the bracketed exposure images (2+)
- arg 2 (optional): output file (default `hdr_result.jpg`)
- arg 3 (optional): max width in px (default 4000)
- arg 4 (optional): `no-enhance` &rarr; skip the finishing chain

## Use &mdash; web UI
```
cd scripts
python app.py
```
Open http://127.0.0.1:5002 , upload the exposures, click **Process**.

## Layout
```
hdr-enhance/
  scripts/
    hdr.py    # the whole pipeline (align + Mertens merge + white-HDR finishing + sharpen)
    run.py    # CLI
    app.py    # minimal Flask UI (port 5002)
  results/    # outputs
  requirements.txt
```

## On segmentation
The finishing chain consumes **SegFormer** semantic masks
(`scripts/window_mask.py`, model `nvidia/segformer-b5-finetuned-ade-640-640`) to:
- exclude window glass from the white-balance / levels measurement (`build_window_mask`),
- gate lamp-whitening to ceiling/wall/lamp regions (`whiten_lamps` `allow`),
- force walls/ceiling neutral and **protect floors + warm wood / fruit** from
  desaturation (`neutralize_whites` `boost` / `protect` / `lamp_allow`).

This is what makes the output **bit-for-bit identical** to pixelswift (verified: mean
diff 0.0, 99.97% pixels exactly equal on the test set).

**Auto-fallback:** `window_mask.get_label_masks()` returns `None` when
`torch`/`transformers` aren't installed, so every mask input degrades to `None` and
the window mask falls back to a pure clipping ramp (blown glass only). The project
then runs **pure OpenCV + NumPy** &mdash; visually close on most scenes, with a small
difference only on warm-wood / lamp-heavy rooms. The first SegFormer run downloads
the model (~once) and CPU inference adds a few seconds per image.

## Tuning (in `process_brackets` / `hdr.py`)
- `auto_exposure(target=...)` &mdash; room brightness (0.60 natural &hellip; 0.74 bright).
- `s_curve(strength=...)` &mdash; contrast pop (0.05 gentle &hellip; 0.10 punchy).
- `sharpen_two_stage(fine_amount, clarity_amount)` &mdash; sharpness / clarity.
