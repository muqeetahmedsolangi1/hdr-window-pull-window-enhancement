"""
hdr-enhance — standalone real-estate HDR (Mertens fusion + white-HDR finishing +
window pull). Pure OpenCV + NumPy; SegFormer segmentation is optional.

Pipeline:
    sort brackets darkest-first
    -> resize to the smallest common size, capped at max_width (INTER_AREA)
    -> align_ecc            (ECC homography alignment to the middle bracket)
    -> cv2.createMergeMertens().process   (the HDR base merge)
    -> white-HDR finishing chain (only when enhance=True):
         white_patch_wb -> auto_levels -> auto_exposure(0.70) -> whiten_lamps
         -> neutralize_whites -> match_neutral_whites -> s_curve(0.05)
    -> sharpen_two_stage    (clarity DAMPED around windows so the boundary
                             doesn't halo)
    -> pull_window_view     (only when pull_windows=True; runs LAST, after sharpen)
    -> uint8 BGR

WINDOW PULL (how the big real-estate editing sites actually do it — researched, not
copied from any other project): the blown-out glass is replaced with the exterior
view from the DARKEST bracket, but ONLY where the base is actually clipped AND the
pixel is the semantic "window" class. So the window FRAME / sash / sill and bright
white appliances are excluded and stay from the bright base. The darker bracket is
lifted to just below room brightness and blended with a Darken (per-pixel min)
operation, which can recover the view but can never darken the frame; the mask edge
is feathered edge-aware (guided filter) to avoid halos and white fringing. Halos are
further prevented by damping the wide-radius "clarity" term inside a band around the
window (local contrast across the high-contrast window boundary is the documented
halo cause).

SEGMENTATION: the finishing chain and the window pull consume SegFormer semantic
masks (window / wall+ceiling / floor / lamp / cabinet). window_mask.get_label_masks()
returns None when torch/transformers are not installed; the finishing chain then
falls back to a pure clipping ramp, and the window pull falls back to a clipping-only
glass mask (less precise on scenes with bright white appliances).

Public API:
    load_exposures(folder)                 -> list[uint8 BGR]
    process_brackets(images, enhance=True, max_width=4000, pull_windows=True)
        -> uint8 BGR
"""
import os
import glob
import cv2
import numpy as np

import window_mask   # SegFormer semantic masks; returns None if torch/transformers absent


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


def light_through_mask(seg_win, darkest, min_p75=0.20, min_area_frac=0.0004):
    """Keep ONLY windows that LIGHT actually comes THROUGH.

    A real window/glass door has an OUTDOOR light source behind it, so it stays
    BRIGHT even in the DARKEST exposure; a solid door / dark wall goes dark. We
    measure the 75th-percentile brightness of each segmented-window blob in the
    darkest bracket and drop the ones that stay dark. Also drops small no-peak
    blobs (a window reflected in a shiny appliance). A "never drop EVERY window"
    safety keeps the single brightest blob if all fail. `darkest` is the darkest
    bracket as float gray [0,1].
    """
    if seg_win is None or darkest is None:
        return seg_win
    h, w = seg_win.shape[:2]
    comp_in = (seg_win > 0.3).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(comp_in)
    out = np.zeros_like(seg_win)
    kept = dropped = 0
    best_score, best_comp = -1.0, None
    reflections = 0
    for i in range(1, n):
        area_frac = st[i, cv2.CC_STAT_AREA] / float(h * w)
        if area_frac < min_area_frac:
            continue
        comp = lab == i
        score = float(np.percentile(darkest[comp], 75))
        peak = float(np.percentile(darkest[comp], 90))
        # REFLECTION / GLINT reject: a real window shows the OUTDOORS (the
        # brightest thing in the scene) so in the darkest bracket it has a bright,
        # often clipped PEAK (p90 ~0.9-1.0). A reflection in a shiny appliance is
        # dim second-hand light and small — no bright peak. Drop small + no-peak.
        if area_frac < 0.012 and peak < 0.80:
            reflections += 1
            dropped += 1
            continue
        if score > best_score:            # remember the brightest candidate
            best_score, best_comp = score, comp
        if score >= min_p75:
            out[comp] = seg_win[comp]
            kept += 1
        else:
            dropped += 1
    if reflections:
        print(f" * light-through gate: dropped {reflections} reflection/glint "
              f"(small + no bright peak — e.g. a window mirrored in an appliance)")
    # SAFETY: never drop EVERY window. If the gate rejected all candidates, keep
    # the single BRIGHTEST one so detection isn't empty.
    if kept == 0 and best_comp is not None:
        out[best_comp] = seg_win[best_comp]
        kept, dropped = 1, max(0, dropped - 1)
        print(f" * light-through gate: every window scored below {min_p75:.2f}; "
              f"kept the brightest (p75={best_score:.2f}) so detection isn't empty")
    elif dropped:
        print(f" * light-through gate: kept {kept} window(s) (light comes "
              f"through), dropped {dropped} solid door / dark surface")
    return out


# ---------------------------------------------------------------- window pull
def _feather_edge(mask, guide_bgr):
    """Edge-aware feather of a soft mask so it snaps to the frame edges (no dark
    fringe, no halo). Uses the guided filter (opencv-contrib `ximgproc`) when that
    build is present — it refines a soft mask into an alpha-matte that follows the
    guidance image's edges — otherwise a plain Gaussian feather. A SMALL radius is
    used on purpose: a large-radius guided filter can itself create halos at a
    strong edge like the window boundary.
    """
    w = mask.shape[1]
    r = max(6, int(w * 0.004))
    src = mask.astype(np.float32)
    try:
        guide = cv2.cvtColor(
            (np.clip(guide_bgr, 0, 1) * 255).astype(np.uint8),
            cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        out = cv2.ximgproc.guidedFilter(guide, src, r, 1e-3)
    except (AttributeError, cv2.error):
        out = cv2.GaussianBlur(src, (0, 0), r * 0.6)
    return np.clip(out, 0, 1)


def _window_band(seg_win, w):
    """A soft band over the window plus a margin. Used to DAMP the wide-radius
    clarity near the high-contrast window boundary — local contrast crossing that
    boundary is the documented cause of glowing/white halo edges."""
    k = max(3, int(w * 0.01) | 1)
    band = cv2.dilate((seg_win > 0.2).astype(np.float32),
                      np.ones((k, k), np.uint8))
    return np.clip(cv2.GaussianBlur(band, (0, 0), k * 0.5), 0, 1)


def _window_protect_mask(seg_win, w):
    """Soft mask over the whole window UNIT (glass + frame/mullions/sash + a small
    margin). Used to KEEP the window's natural Mertens-fused view: the indoor
    finishing/enhance chain blows the window out and haloes it, so we restore the
    window from the raw fused image through this mask and enhance the indoor only.
    Dilated to cover the frame, then feathered for a seamless join with the wall."""
    k = max(3, int(w * 0.010)) | 1
    m = cv2.dilate((seg_win > 0.25).astype(np.float32), np.ones((k, k), np.uint8))
    return np.clip(cv2.GaussianBlur(m, (0, 0), k * 0.7), 0, 1)


def _debug_overlay(base_u8, mask, colour=(0, 200, 0), alpha=0.45):
    """Tint the `mask` region of `base_u8` with `colour` (BGR) so the selection is
    visible on the frontend — used to SHOW which pixels are treated as the window
    (excluded from the indoor enhance / where the crisp view is composited)."""
    m = np.clip(mask, 0, 1)[..., None] * alpha
    tint = np.zeros_like(base_u8, np.float32)
    tint[:] = colour
    return (base_u8.astype(np.float32) * (1 - m) + tint * m).clip(0, 255).astype("uint8")


def _tone_window(pulled, m, target=0.58, saturation=1.10, sharpen=0.15,
                 stretch=0.6):
    """Make the pulled exterior view crisp & natural (NOT over-cooked).

    The view comes from the darkest bracket, so it is correctly exposed for the
    outdoors but dark and flat. A plain gamma lift would flatten the highlights and
    wash out the colour; too-aggressive contrast/saturation/sharpen instead grunges
    the sky and haloes the clouds. So:
      * a GENTLE contrast-stretch on the WINDOW's own histogram (2nd..97th pct),
        BLENDED at `stretch` strength so it restores punch without crushing;
      * a soft gamma lift only if still below room brightness;
      * a mild saturation boost so the blue sky / green foliage pop naturally;
      * a light unsharp for clarity.
    The stats are keyed on the window pixels (`m`) but applied to the whole frame —
    only the window is composited downstream, so the rest is discarded.
    """
    sel = m > 0.5
    if int(sel.sum()) < 50:
        return pulled
    luma = pulled.mean(axis=2)
    lo = float(np.percentile(luma[sel], 2.0))
    hi = float(np.percentile(luma[sel], 97.0))
    if hi - lo > 0.04:
        s = np.clip((pulled - lo) / (hi - lo), 0, 1)
        pulled = pulled * (1 - stretch) + s * stretch            # blended stretch
    med = float(np.median(pulled.mean(axis=2)[sel]))
    if 1e-3 < med < target:                                       # gentle lift only
        gamma = float(np.clip(np.log(target) / np.log(med), 0.65, 1.0))
        pulled = np.clip(pulled, 0, 1) ** gamma
    if saturation != 1.0:                                         # colour pop
        hsv = cv2.cvtColor((np.clip(pulled, 0, 1) * 255).astype(np.uint8),
                           cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        pulled = cv2.cvtColor(hsv.astype(np.uint8),
                              cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    if sharpen > 0:                                               # crispness
        blur = cv2.GaussianBlur(pulled, (0, 0), 1.2)
        pulled = np.clip(pulled + sharpen * (pulled - blur), 0, 1)
    return pulled


def _pick_window_bracket(images, seg_win):
    """Pick the bracket whose WINDOW region shows the CLEAREST exterior view.

    The darkest bracket is not always the prettiest — it can be muddy/noisy, while a
    slightly brighter one holds the most window detail (before it blows out). This is
    the "pick the bracket that contains the most information for the window" step the
    AI editors describe. For each bracket we measure, inside the window region, the
    amount of visible detail (gradient magnitude) over the pixels that are WELL-EXPOSED
    there (not crushed to black, not blown to white). Most well-exposed detail wins.
    `images` is sorted darkest->brightest; returns the chosen index.
    """
    reg = seg_win > 0.3
    n_reg = int(reg.sum())
    if n_reg < 50:
        return 0
    best_i, best_score = 0, -1.0
    for i, im in enumerate(images):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        good = reg & (g > 0.06) & (g < 0.90)          # well-exposed window pixels
        if int(good.sum()) < 50:
            continue
        score = float(grad[good].sum()) / n_reg       # visible detail per window area
        if score > best_score:
            best_score, best_i = score, i
    return best_i


def pull_window_view(base, view_u8, dark_u8, seg_win, clip_lo=0.72,
                     lit_lo=0.05, lit_hi=0.11, chroma_lo=0.04, chroma_hi=0.12):
    """Composite the exterior view into the blown glass, using TWO decoupled brackets.

      * `view_u8` — the bracket with the CLEAREST window view (from
        _pick_window_bracket). It supplies the actual exterior CONTENT, so the view is
        crisp and never washed out even when the darkest frame is muddy/dark.
      * `dark_u8` — the DARKEST bracket, used ONLY as the LIT reference: the window's
        frame/casing/mullions are near-black there, so the lit gate excludes them and
        keeps them from the bright base (no dark frame, no dark ring/box).

    Glass mask = SEMANTIC (SegFormer window `seg_win`) x BLOWN (base is clipped) x LIT
    (bright-or-colourful in the darkest bracket). The view is FULL-opacity inside the
    glass (feather only the boundary) so it is never diluted with the white base. The
    view bracket is toned by `_tone_window`, then blended with Darken (per-pixel min)
    so it can never darken a brighter frame.

    `base` is float BGR [0,1] (finished + sharpened); `view_u8`/`dark_u8` are aligned
    uint8 BGR brackets; `seg_win` is the soft window mask [0,1].
    """
    base = np.clip(base, 0, 1).astype(np.float32)
    base_luma = base.mean(axis=2)
    view = view_u8.astype(np.float32) / 255.0
    dark = dark_u8.astype(np.float32) / 255.0
    dark_luma = dark.mean(axis=2)

    blown = base_luma > clip_lo
    region = (np.clip(seg_win, 0, 1) > 0.35) & blown       # semantic ∩ blown glass box
    if not region.any():
        return base                      # no see-through blown glass found

    # LIT from the DARKEST bracket: a window pixel is "outdoors seen through glass" if
    # there it is either BRIGHT (sky, sunny patio) OR COLOURFUL (blue pool/chairs,
    # green palms — dark but saturated). The frame/casing/mullions are near-black AND
    # grey, so they fail both. Used as a BINARY gate (not a soft opacity): the view
    # gets FULL opacity so it is never diluted with the white base (that dilution is
    # the milky/washed haze), while the frame is fully excluded (no dark box).
    chroma = dark.max(axis=2) - dark.min(axis=2)
    bright = np.clip((dark_luma - lit_lo) / max(lit_hi - lit_lo, 1e-3), 0, 1)
    colourful = np.clip((chroma - chroma_lo) / max(chroma_hi - chroma_lo, 1e-3), 0, 1)
    lit = np.maximum(bright, colourful)
    glass = (region & (lit > 0.4)).astype(np.float32)
    if float(glass.max()) < 1e-3:
        glass = region.astype(np.float32)               # everything scored low: fill it
    # CLOSE small holes so the blown-white base can't peek through the view (white
    # speckle/shade). Kernel kept small so thin white mullions survive.
    kf = max(3, int(base.shape[1] * 0.004)) | 1
    glass = cv2.morphologyEx(glass, cv2.MORPH_CLOSE, np.ones((kf, kf), np.uint8))
    glass *= region.astype(np.float32)                  # stay inside the glass box

    pulled = _tone_window(view, region.astype(np.float32))
    darkened = np.minimum(base, pulled)              # Darken: never darken the frame
    mf = _feather_edge(glass, base)[..., None]        # soften only the boundary
    return np.clip(base * (1 - mf) + darkened * mf, 0, 1)


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


def sharpen_two_stage(img, fine_amount=0.9, clarity_amount=0.2, protect=None):
    """Fine unsharp mask for edge crispness + wide-radius local contrast
    (the Lightroom 'Clarity' effect).

    `protect` (soft [0,1] window band): the wide-radius clarity term is damped
    there — clarity crossing the high-contrast window boundary is the documented
    cause of glowing/white halo edges, so we turn it down around windows. The fine
    (1px) unsharp is left alone to keep frame edges crisp.
    """
    fine = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = np.clip(img + fine_amount * (img - fine), 0, 1)
    wide = cv2.GaussianBlur(img, (0, 0), 15)
    ca = clarity_amount
    if protect is not None:
        ca = clarity_amount * (1.0 - protect)[..., None]
    return np.clip(img + ca * (img - wide), 0, 1)


# ---------------------------------------------------------------- pipeline
def process_brackets(images, enhance=True, max_width=4000, pull_windows=False,
                     return_debug=False):
    """Merge the bracketed exposures (Mertens fusion), apply the white-HDR finishing
    chain, and pull the windows. Returns uint8 BGR. Pure OpenCV + NumPy.

    enhance=False     -> plain Mertens fusion + two-stage sharpen, with NO white
                         balance / levels / exposure / lamp-whitening / neutralize
                         / S-curve.
    pull_windows=True -> after sharpening, composite the exterior view from the
                         darkest bracket into the blown glass only (semantic window
                         INTERSECT clipping), Darken/min blend with an edge-aware
                         feather, so the frame stays bright and the boundary doesn't
                         halo. Needs the SegFormer window mask for the precise
                         glass-only result; without torch/transformers it falls back
                         to a clipping-only glass mask.
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

    img = fused.astype(np.float32)
    mid = images[len(images) // 2]

    dbg = {}
    if return_debug:                       # raw Mertens fusion the pipeline starts from
        dbg["fused"] = (np.clip(fused, 0, 1) * 255).astype("uint8")

    # ---- WINDOW DETECTION (needed by the finishing exclusion AND the window pull)
    masks = None
    seg_win = None
    if enhance or pull_windows:
        # WINDOW segmentation from the SUPER-DARKEST bracket (lowest hallucination:
        # only real windows still show outdoor light there), notwindow subtracted.
        # get_label_masks returns None if torch/transformers are unavailable.
        dmasks = window_mask.get_label_masks(images[0])
        seg_win = dmasks["window"] if dmasks else None
        if seg_win is not None and dmasks.get("notwindow") is not None:
            nwd = cv2.dilate(dmasks["notwindow"],
                             np.ones((max(3, int(w * 0.004) | 1),) * 2, np.uint8))
            seg_win = np.clip(seg_win - nwd, 0, 1)
        if enhance:
            # interior masks (mid frame): wall/ceiling, floor, lamp, cabinet.
            masks = window_mask.get_label_masks(mid)
            # SAFETY fallback to the mid-frame window mask if the darkest found none.
            if (seg_win is None or float((seg_win > 0.3).sum()) < 1) \
                    and masks and masks.get("window") is not None:
                seg_win = masks["window"].copy()
                if masks.get("notwindow") is not None:
                    nwm = cv2.dilate(masks["notwindow"],
                                     np.ones((max(3, int(w * 0.004) | 1),) * 2, np.uint8))
                    seg_win = np.clip(seg_win - nwm, 0, 1)
        # LIGHT-THROUGH gate: keep only windows light actually comes through.
        if seg_win is not None:
            darkest = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            seg_win = light_through_mask(seg_win, darkest)

    if enhance:
        wmask = build_window_mask(mid, seg_win)

        # where lamp-whitening is permitted: ceiling/wall/lamp, minus cabinets
        # and other warm wood surfaces (so sunlit glints on wood keep color)
        lamp_allow = None
        if masks:
            def _m(name):
                return masks[name] if masks.get(name) is not None \
                    else np.zeros(mid.shape[:2], np.float32)
            allow = np.clip(_m("wallceil") + _m("lamp") - _m("cabinet"), 0, 1)
            kk = max(3, int(mid.shape[1] * 0.01) | 1)
            lamp_allow = cv2.GaussianBlur(
                cv2.dilate(allow, np.ones((kk, kk), np.uint8)), (kk, kk), 0)

        # interior finishing — windows excluded from WB and levels measurement.
        # Brightness FIRST, color cleanup AFTER: the lamp/wall neutralizers key on
        # bright pixels, so they only see the casts once the image is at its final
        # brightness.
        img = white_patch_wb(img, exclude=wmask)
        img = auto_levels(img, exclude=wmask)
        img = auto_exposure(img, target=0.70)   # AutoHDR median ~189/255
        img = whiten_lamps(img, allow=lamp_allow)
        img = neutralize_whites(
            img,
            boost=masks["wallceil"] if masks else None,
            protect=masks["floor"] if masks else None,
            lamp_allow=lamp_allow,
        )
        img = match_neutral_whites(img)
        img = s_curve(img, strength=0.05)        # measured: gentle contrast

    have_window = seg_win is not None and float(seg_win.max()) > 1e-3

    # WINDOW = crisp single-bracket view + kept out of the enhance.
    #   (1) Mertens blends ALL brackets for the window and WASHES IT OUT (the blown
    #       brighter brackets dilute the crisp darker-bracket view -> low contrast).
    #       So we composite the CLEAREST single bracket's view (glass only) into the
    #       fused image, lightly lifted to sit naturally.
    #   (2) The finishing/enhance chain would then blow that window out and halo it
    #       (white halos on water, washed trees), so we HDR the INDOOR ONLY and
    #       restore the window — glass AND frame — from that improved fused image.
    if enhance and have_window and not pull_windows:
        vi = _pick_window_bracket(images, seg_win)
        if vi != 0:
            print(f" * window: using bracket #{vi} of {len(images)} (darkest=0) "
                  f"for a crisp view — the Mertens blend washed it out")
        raw = images[vi].astype(np.float32) / 255.0
        vlum = raw.mean(axis=2)
        vchr = raw.max(axis=2) - raw.min(axis=2)
        # keep the bright fused FRAME/mullions: bring the crisp view in ONLY where the
        # chosen bracket is the actual see-through view (BRIGHT or COLOURFUL). The
        # window's frame/mullions are dark in that (darker) bracket, so they score ~0
        # and the fused BRIGHT frame shows there — view gets crisp, frame stays bright.
        litw = np.maximum(np.clip((vlum - 0.06) / 0.10, 0, 1),
                          np.clip((vchr - 0.05) / 0.08, 0, 1))
        sel = seg_win > 0.3
        med = float(np.median(vlum[sel])) if sel.any() else 0.0
        view = raw
        if 1e-3 < med < 0.58:                          # gentle lift, keep contrast
            view = np.clip(raw, 0, 1) ** float(np.clip(
                np.log(0.58) / np.log(med), 0.6, 1.0))
        gm = _feather_edge(np.clip(seg_win, 0, 1).astype(np.float32) * litw, fused)[..., None]
        fused = fused * (1 - gm) + view * gm           # crisp view into the glass only
        wm = _window_protect_mask(seg_win, w)[..., None]
        img = img * (1 - wm) + fused * wm
        if return_debug:
            base = dbg.get("fused")
            # GREEN = window region kept out of the indoor enhance (the exclusion);
            # BLUE  = where the crisp single-bracket view is composited (the glass).
            dbg["window_excluded"] = _debug_overlay(base, wm[..., 0], (0, 200, 0))
            dbg["window_glass"] = _debug_overlay(base, gm[..., 0], (230, 120, 0))
            dbg["window_bracket"] = vi     # which bracket supplied the window view

    # SHARPEN — clarity damped in a band around the window so the high-contrast
    # window boundary can't halo.
    protect = _window_band(seg_win, w) if have_window else None
    img = sharpen_two_stage(img, protect=protect)

    # OPTIONAL experimental darker-bracket window pull (off by default — the fused
    # window above is cleaner). Enable with pull_windows=True.
    if pull_windows and have_window:
        vi = _pick_window_bracket(images, seg_win)
        if vi != 0:
            print(f" * window view: pulled from bracket #{vi} of {len(images)} "
                  f"(darkest=0) — it had the clearest view")
        img = pull_window_view(img, images[vi], images[0], seg_win)

    result = (img * 255).clip(0, 255).astype("uint8")
    if return_debug:
        # if no window was detected, still expose the window overlays (= no selection)
        dbg.setdefault("window_excluded", dbg.get("fused"))
        dbg.setdefault("window_glass", dbg.get("fused"))
        return result, dbg
    return result
