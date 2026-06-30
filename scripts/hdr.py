"""
hdr-enhance — standalone real-estate HDR (the pixelswift "HDR enhance" checkbox).

This is a SELF-CONTAINED port of the FIRST checkbox of the pixelswift project
("HDR enhance — calibrated color/tone finishing"), i.e. pipeline.process_brackets
run with enhance=True and pull_windows=False. It does NOT pull/segment windows and
does NOT call any AI/generative step.

Pipeline (identical order to pixelswift):
    sort brackets darkest-first
    -> resize to the smallest common size, capped at max_width (INTER_AREA)
    -> align_ecc            (ECC homography alignment to the middle bracket)
    -> cv2.createMergeMertens().process   (the HDR base merge)
    -> white-HDR finishing chain (only when enhance=True):
         white_patch_wb -> auto_levels -> auto_exposure(0.70) -> whiten_lamps
         -> neutralize_whites -> match_neutral_whites -> s_curve(0.05)
    -> sharpen_two_stage    (ALWAYS, runs last)
    -> uint8 BGR

Pure OpenCV + NumPy. Fully self-contained (no imports from any other project).

NOTE on segmentation: in pixelswift the finishing chain optionally consumes
SegFormer semantic masks (wall/ceiling/floor/lamp/cabinet) to *refine* where
lamp-whitening / white-neutralization apply, and to exclude window glass from the
white-balance / levels measurement. The pixelswift code is written so every one of
those mask inputs degrades to None when segmentation is unavailable, and the window
mask falls back to a pure clipping ramp (blown glass only). This port uses exactly
that no-segmentation path — so it is the same maths, with no torch/transformers
dependency. (See README for how to re-add SegFormer for exact parity.)

Public API:
    load_exposures(folder)                 -> list[uint8 BGR]
    process_brackets(images, enhance=True, max_width=4000) -> uint8 BGR
"""
import os
import glob
import cv2
import numpy as np


# ---------------------------------------------------------------- IO
_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff",
         "*.JPG", "*.JPEG", "*.PNG", "*.TIF", "*.TIFF")


def load_exposures(folder):
    """Read all bracketed exposure images from a folder (any common format)."""
    files = []
    for e in _EXTS:
        files += glob.glob(os.path.join(folder, e))
    files = sorted(set(os.path.normcase(f) for f in files))
    imgs = [cv2.imread(f) for f in files]
    imgs = [im for im in imgs if im is not None]
    if len(imgs) < 2:
        raise ValueError(f"need >= 2 exposures in {folder!r}, found {len(imgs)}")
    return imgs


# ---------------------------------------------------------------- alignment
def align_ecc(images, work_width=1000):
    """Align all brackets to the middle one with a full homography.

    ECC is estimated on a downscaled, histogram-equalized copy (robust to
    exposure differences, fast), then the homography is rescaled and
    applied at full resolution. Falls back to the unwarped image when a
    bracket fails to converge.
    """
    ref = len(images) // 2
    h, w = images[0].shape[:2]
    scale = min(1.0, work_width / w)
    small = [
        cv2.equalizeHist(
            cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None,
                       fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        )
        for im in images
    ]
    S = np.diag([scale, scale, 1.0])
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)

    aligned = []
    for i, im in enumerate(images):
        if i == ref:
            aligned.append(im)
            continue
        warp = np.eye(3, dtype=np.float32)
        try:
            cv2.findTransformECC(small[ref], small[i], warp,
                                 cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
            H = (np.linalg.inv(S) @ warp.astype(np.float64) @ S).astype(np.float32)
            aligned.append(cv2.warpPerspective(
                im, H, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE))
        except cv2.error:
            aligned.append(im)
    return aligned


# ---------------------------------------------------------------- window mask
def build_window_mask(mid, seg_mask=None):
    """Soft mask of blown window glass (used to EXCLUDE the window from the
    white-balance / levels measurement, so the white point anchors on the WALLS,
    not the brighter window).

    Gated by actual clipping in the mid exposure (>= 235) so a wall or frame can
    never be touched. With no segmentation `seg_mask` is None and the mask is the
    pure clipping ramp; a wrong label cannot create an artifact because non-clipped
    pixels stay at zero.
    """
    mid_gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ramp = np.clip((mid_gray - 235.0) / 20.0, 0, 1)
    if seg_mask is not None:
        ramp = ramp * (0.35 + 0.65 * seg_mask)
    k = max(31, int(mid_gray.shape[1] * 0.008) | 1)
    return cv2.GaussianBlur(ramp, (k, k), 0)


# ---------------------------------------------------------------- finishing
def white_patch_wb(img, percentile=92.0, strength=1.0, exclude=None):
    """Neutralize color cast using the brightest non-clipped pixels.

    Real estate interiors almost always have white walls/ceilings, so the
    bright regions are a reliable neutral reference — unlike gray-world,
    which is skewed by wood floors and furniture.

    `exclude` (window mask) is essential: without it the measurement is
    dominated by window glass showing bluish daylight, and "correcting"
    that blue to neutral turns the whole interior golden.
    """
    luma = img.mean(axis=2)
    valid = luma < 0.98
    if exclude is not None:
        valid &= exclude < 0.5
    vals = luma[valid]
    if vals.size < 1000:
        return img
    ref = img[valid & (luma >= np.percentile(vals, percentile))]
    if len(ref) < 100:  # everything clipped — nothing to measure
        return img
    means = ref.mean(axis=0)
    gains = means.max() / np.maximum(means, 1e-6)
    gains = 1 + (gains - 1) * strength
    return np.clip(img * gains, 0, 1)


def auto_levels(img, black_pct=0.4, white_pct=99.7, max_black=0.25,
                exclude=None):
    """Anchor the black and white points (Lightroom-style auto levels).

    Brightening alone leaves the image hazy: shadows that should be black
    turn gray. Stretching so the darkest percentile maps to 0 and the
    brightest to 1 restores contrast and makes white walls truly white.
    The same stretch is applied to all channels so colors don't shift.

    `exclude` (window mask): window pixels are left out of the percentile
    measurement so the white point anchors on the WALLS — otherwise the
    windows are always the brightest pixels and the walls stay gray.
    """
    luma = img.mean(axis=2)
    sample = luma if exclude is None else luma[exclude < 0.5]
    if sample.size < 1000:
        sample = luma
    lo = min(float(np.percentile(sample, black_pct)), max_black)
    hi = float(np.percentile(sample, white_pct))
    if hi - lo < 0.1:
        return img
    # measured from AutoHDR outputs: blacks sit at ~17/255 (soft, not
    # crushed) and whites at ~235/255 (bright but never blasted); the
    # later gamma brighten lifts the floor further, hence 0.02 here
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return out * (0.91 - 0.02) + 0.02


def auto_exposure(img, target=0.60):
    """Brighten until the median luminance hits the target (gamma-based).

    Gamma lifts midtones without clipping the highlights. Only ever
    brightens, never darkens.
    """
    luma = img.mean(axis=2)
    med = max(float(np.median(luma)), 1e-3)
    if med >= target:
        return img
    gamma = np.clip(np.log(target) / np.log(med), 0.45, 1.0)
    return np.clip(img, 0, 1) ** gamma


def whiten_lamps(img, strength=1.0, max_area_frac=0.015, allow=None):
    """Turn warm light sources white (the AutoHDR rendering).

    Finds small, bright, warm blobs — bulbs, lamp shades, recessed-can
    glow pools — and pulls their color to neutral in LAB. Large warm
    areas (wood floors, furniture) are excluded by the component-size
    filter.

    `allow` (segmented ceiling/wall/lamp region, cabinets removed) gates
    WHERE whitening may happen. Without segmentation `allow` is None and
    only the component-size filter limits the effect (the classic behavior).
    """
    lab = cv2.cvtColor((img * 255).astype(np.uint8),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] / 255.0
    b = lab[..., 2] - 128.0
    warm = ((L > 0.65) & (b > 8)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(warm)
    if n < 2:
        return img
    h, w = warm.shape
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < h * w * max_area_frac
    keep = small[labels].astype(np.float32)
    if allow is not None:
        keep *= allow
    keep = cv2.GaussianBlur(keep, (0, 0), max(4.0, w * 0.002)) * strength
    lab[..., 1] = (lab[..., 1] - 128.0) * (1 - keep) + 128.0
    lab[..., 2] = (lab[..., 2] - 128.0) * (1 - keep) + 128.0
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out.astype(np.float32) / 255.0


def neutralize_whites(img, strength=1.0, boost=None, protect=None,
                      lamp_allow=None):
    """Remove local warm/cool casts from surfaces that should be white.

    Mixed lighting (tungsten bulbs + daylight) tints walls and ceilings
    differently across the frame; one global WB can't fix both. Pulled
    toward neutral in LAB:
      * bright, low-chroma surfaces (whites with a mild cast);
      * `boost` mask (segmented walls+ceiling): forced neutral even when
        warm light tinted them heavily;
      * very bright pixels — lamps/bulbs rendered white not orange; this
        term is gated by `lamp_allow` (ceiling/wall/lamp region).
    `protect` (segmented floor/rug) is exempted so floors keep color.
    With no segmentation boost/protect/lamp_allow are all None and only the
    bright low-chroma term applies (the classic behavior).
    """
    lab = cv2.cvtColor((img * 255).astype(np.uint8),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] / 255.0
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    w_walls = (np.clip((L - 0.70) / 0.12, 0, 1)
               * np.clip((22.0 - chroma) / 8.0, 0, 1))
    if boost is not None:
        w_seg = (boost * np.clip((L - 0.55) / 0.20, 0, 1)
                 * np.clip((35.0 - chroma) / 10.0, 0, 1))
        w_walls = np.maximum(w_walls, w_seg)
    if protect is not None:
        w_walls = w_walls * (1 - protect)
    w_lamps = np.clip((L - 0.85) / 0.10, 0, 1)
    if lamp_allow is not None:
        # off the ceiling/wall/lamp, only near-white (L>0.95) pixels are
        # neutralized; saturated bright wood/metal keeps its color
        w_lamps *= np.maximum(lamp_allow, np.clip((L - 0.95) / 0.05, 0, 1))
    w = cv2.GaussianBlur(np.maximum(w_walls, w_lamps), (0, 0), 8) * strength
    lab[..., 1] = a * (1 - w) + 128.0
    lab[..., 2] = b * (1 - w) + 128.0
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out.astype(np.float32) / 255.0


def match_neutral_whites(img, strength=0.85):
    """Global temperature/tint trim: shift LAB a/b so the image's bright
    pixels land on neutral (a=b=128). This is the measured difference
    between our output and AutoHDR's — their whites sit at exactly
    128/128 while warm casts leave ours a few points yellow."""
    lab = cv2.cvtColor((img * 255).astype(np.uint8),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    bright = lab[..., 0] > 0.78 * 255
    if bright.sum() < 1000:
        return img
    lab[..., 1] += (128.0 - float(lab[..., 1][bright].mean())) * strength
    lab[..., 2] += (128.0 - float(lab[..., 2][bright].mean())) * strength
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out.astype(np.float32) / 255.0


def s_curve(img, strength=0.08):
    """True S-curve: deepens shadows, lifts highlights."""
    return np.clip(img + strength * np.sin(2 * np.pi * (img - 0.5)), 0, 1)


def sharpen_two_stage(img, fine_amount=0.9, clarity_amount=0.2):
    """Fine unsharp mask for edge crispness + wide-radius local contrast
    (the Lightroom 'Clarity' effect). Runs last in the pipeline."""
    fine = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = np.clip(img + fine_amount * (img - fine), 0, 1)
    wide = cv2.GaussianBlur(img, (0, 0), 15)
    return np.clip(img + clarity_amount * (img - wide), 0, 1)


# ---------------------------------------------------------------- pipeline
def process_brackets(images, enhance=True, max_width=4000):
    """The pixelswift first-checkbox HDR path (enhance=True, pull_windows=False),
    pure OpenCV + NumPy. Merges the bracketed exposures with Mertens fusion and
    applies the calibrated 'white HDR' finishing chain. Returns uint8 BGR.

    enhance=False reproduces the checkbox-OFF behavior: plain Mertens fusion +
    two-stage sharpen, with NO white balance / levels / exposure / lamp-whitening
    / neutralization / S-curve.
    """
    if len(images) < 2:
        raise ValueError("need at least 2 bracketed exposures")

    # darkest first — upload order is unknown
    images = sorted(images, key=lambda im: im.mean())

    # alignment needs equal sizes; resize to the smallest frame, capped at max_width
    h = min(im.shape[0] for im in images)
    w = min(im.shape[1] for im in images)
    if w > max_width:
        h = int(h * max_width / w)
        w = max_width
    images = [
        im if im.shape[:2] == (h, w)
        else cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        for im in images
    ]

    images = align_ecc(images)
    fused = cv2.createMergeMertens().process(images)

    mid = images[len(images) // 2]
    # No segmentation: window mask is the pure clipping ramp (blown glass only),
    # used to exclude the window from the WB / levels measurement.
    wmask = build_window_mask(mid)

    img = fused.astype(np.float32)
    if enhance:
        # interior finishing — windows excluded from WB and levels measurement.
        # Brightness FIRST, color cleanup AFTER: the lamp/wall neutralizers key on
        # bright pixels, so they only see the casts once the image is at its final
        # brightness.
        img = white_patch_wb(img, exclude=wmask)
        img = auto_levels(img, exclude=wmask)
        img = auto_exposure(img, target=0.70)   # AutoHDR median ~189/255
        img = whiten_lamps(img)
        img = neutralize_whites(img)
        img = match_neutral_whites(img)
        img = s_curve(img, strength=0.05)        # measured: gentle contrast

    img = sharpen_two_stage(img)                 # ALWAYS, runs last
    return (img * 255).clip(0, 255).astype("uint8")
