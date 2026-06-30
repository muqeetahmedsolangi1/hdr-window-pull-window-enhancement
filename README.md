# hdr-enhance

A **standalone**, self-contained port of the **first checkbox** of the pixelswift
project &mdash; *"HDR enhance &mdash; calibrated color/tone finishing"* &mdash; i.e.
`pipeline.process_brackets(..., enhance=True, pull_windows=False)`.

It merges a set of **bracketed exposures** with classic **Mertens exposure fusion**
and then applies the calibrated **"white HDR"** finishing chain to produce the bright,
clean, neutral-white real-estate look. No window pull, no segmentation by default, no
AI/generative step.

Pure **OpenCV + NumPy**. Nothing is imported from any other project.

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

## On segmentation (faithfulness note)
In pixelswift the finishing chain *optionally* consumes SegFormer semantic masks
(wall / ceiling / floor / lamp / cabinet) to **refine** where lamp-whitening and
white-neutralization apply, and to exclude window glass from the white-balance /
levels measurement. The pixelswift code is written so **every one of those mask
inputs degrades to `None`** when segmentation is unavailable, and the window mask
falls back to a **pure clipping ramp** (blown glass only).

This project uses exactly that no-segmentation path &mdash; the **same maths**, with no
`torch` / `transformers` dependency. For the vast majority of scenes the result is
visually identical; the seg masks mainly help on rooms with large warm-wood / floor
areas (they stop lamp-whitening from touching sunlit wood glints and keep floors from
being neutralized). To get **exact parity** you would add a `get_label_masks(frame)`
that returns `{wallceil, floor, lamp, cabinet, window, notwindow}` and pass those
into `build_window_mask` (`seg_mask`), `whiten_lamps` (`allow`) and
`neutralize_whites` (`boost`/`protect`/`lamp_allow`) &mdash; the function signatures
already accept them.

## Tuning (in `process_brackets` / `hdr.py`)
- `auto_exposure(target=...)` &mdash; room brightness (0.60 natural &hellip; 0.74 bright).
- `s_curve(strength=...)` &mdash; contrast pop (0.05 gentle &hellip; 0.10 punchy).
- `sharpen_two_stage(fine_amount, clarity_amount)` &mdash; sharpness / clarity.
