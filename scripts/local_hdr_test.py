"""
LOCAL HDR TEST — pure HDR only: align + Mertens fuse the brackets, then run the
finishing chain from clear_seg_window_pull.py's CELL 7 (Colab), plus ONLY the
3 MUDDY-LOOK fixes (below). NO segmentation, NO window mask, NO glass
compositing — just the tone-mapping chain on the whole image.

MUDDY-LOOK FIX (the only deviation from the Colab CELL 7 chain — calibrated on
a real photo against an AutoHDR reference; no shadow-pocket lift/rescue here):
  1) _expose target 0.70 -> 0.55 — 0.70 over-lifted an already-balanced fusion
     and flattened/washed the frame
  2) _neutralize also de-tints shadows/midtones (tight chroma gate — real
     wood/fabric colour never touched), removing the warm dirty cast on dark
     surfaces
  3) _sharpen's clarity damped in shadows — uniform clarity on dark pixels is
     what reads as "muddy/gritty"

Differences vs the Colab pipeline: no glass mask (the `exclude=` parameter of
_wb/_levels is unused) and no darker-image glass composite.

Asks for (typed at the prompt, no GUI):
  a folder of exposure BRACKETS -> aligned (ECC) + Mertens FUSED locally

Run:  python local_hdr_test.py
"""
import os
import glob
import numpy as np
import cv2

_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff",
         "*.JPG", "*.JPEG", "*.PNG", "*.TIF", "*.TIFF")


def _ask(prompt):
    val = input(prompt).strip().strip('"')
    if not val:
        raise SystemExit("cancelled — no input given")
    return val


def _load(path):
    im = cv2.imread(path)
    if im is None:
        raise RuntimeError(f"could not read '{path}'")
    return im


def _align(imgs, ref_img, Wd, H, wwork=1000):
    """ECC-align every image in `imgs` to `ref_img` (same recipe as CELL 6)."""
    s = min(1.0, wwork / Wd)
    g_ref = cv2.equalizeHist(cv2.resize(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), None,
                                        fx=s, fy=s, interpolation=cv2.INTER_AREA))
    S = np.diag([s, s, 1.0])
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    out = []
    for im in imgs:
        g_im = cv2.equalizeHist(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None,
                                           fx=s, fy=s, interpolation=cv2.INTER_AREA))
        warp = np.eye(3, dtype=np.float32)
        try:
            cv2.findTransformECC(g_ref, g_im, warp, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
            Hm = (np.linalg.inv(S) @ warp.astype(np.float64) @ S).astype(np.float32)
            out.append(cv2.warpPerspective(im, Hm, (Wd, H),
                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_REPLICATE))
        except cv2.error:
            out.append(im)
    return out


# -------------- HDR CHAIN — identical to clear_seg_window_pull.py CELL 7 -------
def _wb(img, pct=92, exclude=None):
    luma = img.mean(2); valid = luma < 0.98
    if exclude is not None: valid &= exclude < 0.5
    v = luma[valid]
    if v.size < 1000: return img
    ref = img[valid & (luma >= np.percentile(v, pct))]
    if len(ref) < 100: return img
    g = ref.mean(0); g = g.max() / np.maximum(g, 1e-6)
    return np.clip(img * g, 0, 1)


def _levels(img, blk=0.4, wht=99.7, maxblk=0.25, exclude=None):
    luma = img.mean(2); s2 = luma if exclude is None else luma[exclude < 0.5]
    if s2.size < 1000: s2 = luma
    lo = min(float(np.percentile(s2, blk)), maxblk); hi = float(np.percentile(s2, wht))
    if hi - lo < 0.1: return img
    return np.clip((img - lo) / (hi - lo), 0, 1) * (0.91 - 0.02) + 0.02


def _shadow_fill(fused, brightest, T=0.18, p=1.5, max_fill=0.30, feather=4):
    """FUSION-TIME deep-shadow fill from the BRIGHTEST bracket.

    Root cause (measured on the real photo): the table's bottom-shelf pocket
    is clearly visible in the brightest bracket (EV+2.9, luma 0.396) but the
    Mertens result lands at 0.049 — because that bracket is ~91% blown, its
    well-exposedness weight is regionally ~0 at the coarse pyramid levels, and
    the multi-scale smoothing suppresses it even at the few pixels where it is
    actually the BEST-exposed frame. Boosting cv2.createMergeMertens's
    exposure_weight does nothing (verified: 0.049 -> 0.051 at ew=5).

    So the fill happens explicitly: wherever the fused luma is CRUSHED
    (< T=0.18 — walls are 0.43, wood 0.24, pot 0.30, all untouched), blend in
    the brightest bracket's real pixels, at most `max_fill` of the way. The
    weight is a smooth function of the fused LUMINANCE itself (not an edge/
    structure detector), so it cannot produce the silhouette-halo artifacts
    the earlier pocket-rescue attempts did. Calibrated against AutoHDR:
    pocket 0.222 (A:0.234), chair 0.232 (A:0.233), side table 0.450 (A:0.475).
    """
    fl = fused.astype(np.float32).mean(2) / 255
    w = np.clip((T - fl) / T, 0, 1) ** p
    w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), feather) * max_fill
    w3 = w[..., None]
    out = fused.astype(np.float32) * (1 - w3) + brightest.astype(np.float32) * w3
    return out.clip(0, 255).astype("uint8")


def _bright_ramp(img):
    """Soft mask of already-bright/clipped pixels (the blown window) — the
    no-segmentation stand-in for the Colab glass mask, same trick as
    hdr.py's build_window_mask fallback. Used to EXCLUDE the window from
    measurements and to SHIELD it from the interior brightness lifts, so the
    interior can be pushed to AutoHDR levels without blasting the view."""
    luma = img.mean(2)
    ramp = np.clip((luma - 0.80) / 0.12, 0, 1)
    k = max(31, int(img.shape[1] * 0.008) | 1)
    return cv2.GaussianBlur(ramp, (k, k), 0)


def _expose(img, target=0.68, exclude=None):
    # Measured against AutoHDR on the real photo: its INTERIOR sits much
    # brighter (walls ~0.75) than a 0.55 target produces, but its WINDOW is
    # darker than ours — so the median is measured on NON-bright pixels only
    # (exclude = _bright_ramp) and the gamma is FADED OUT over the bright
    # ramp, brightening the room without pushing the already-blown glass.
    luma = img.mean(2)
    sel = luma if exclude is None else luma[exclude < 0.5]
    if sel.size < 1000: sel = luma
    m = max(float(np.median(sel)), 1e-3)
    if m >= target:
        return img
    gamma = float(np.clip(np.log(target) / np.log(m), 0.45, 1.0))
    lifted = np.clip(img, 0, 1) ** gamma
    if exclude is None:
        return lifted
    ex = exclude[..., None]
    return lifted * (1 - ex) + np.clip(img, 0, 1) * ex


def _neutralize(img):
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]/255; a = lab[..., 1]-128; b = lab[..., 2]-128
    ch = np.sqrt(a*a+b*b)
    wgt = np.clip((L-0.70)/0.12, 0, 1) * np.clip((22-ch)/8, 0, 1)
    wgt = np.maximum(wgt, np.clip((L-0.85)/0.10, 0, 1))
    # MUDDY FIX 2/3: the terms above SKIP everything below L=0.70, so a warm/
    # cool cast on a dark table or shadowed wall never got cleaned — that tint
    # is the colour half of the muddy look. Tight chroma gate (<14 vs 22) so
    # real colour (wood grain, teal fabric) is never touched, only near-neutral
    # surfaces tinted by mixed lighting; capped at 0.5 strength so shadows are
    # de-tinted, not flattened.
    shadow_wgt = np.clip((14-ch)/6, 0, 1) * np.clip((L-0.04)/0.05, 0, 1) * 0.5
    wgt = np.maximum(wgt, shadow_wgt)
    # HALO-BAND FIX: tungsten shading on a white ceiling/wall sits at mid-luma
    # (L 0.5-0.7) with chroma 14-24 — the GAP between the two terms above — so
    # after the open ceiling is whitened, its shaded junction strip stays tan
    # (measured: open ceiling b+2 vs band b+18 = the visible halo). Catch
    # near-neutral pure-YELLOW pixels only: the a-channel gate spares green
    # walls (a<=-5) and pink fabric / orange wood (a>=+4). Strength is
    # a-dependent: full 0.9 on paint shading (a<=-1.5, which must end WHITE)
    # but only 0.55 on warm-tan fabric (a>=+0.5, curtains stay beige).
    ga = np.clip((a+5)/3, 0, 1) * np.clip((4-a)/4, 0, 1)
    strength = 0.55 + 0.35*np.clip((0.5-a)/2, 0, 1)
    yellow_wgt = (ga * np.clip((b-4)/6, 0, 1) * np.clip((28-ch)/8, 0, 1)
                  * np.clip((L-0.45)/0.12, 0, 1) * strength)
    # BULB-GLOW path (kitchen pendant): a warm bulb's glow on a white ceiling
    # is slightly a-POSITIVE (a +1..+3), which the a-dependent strength above
    # treats as fabric and barely touches (measured: glow ended b+10 vs
    # AutoHDR b+1). Illumination cast is b-dominated with small a RELATIVE to
    # its chroma — (ch-2a) sits at 7..11 for the glow but >=13 for real tan
    # curtain folds — so that axis separates them where a and ch alone cannot.
    sc = ch - 2*a
    glow_wgt = (np.clip((6-a)/4, 0, 1) * np.clip((13.5-sc)/3.5, 0, 1)
                * np.clip((b-4)/6, 0, 1) * np.clip((L-0.45)/0.12, 0, 1) * 0.9)
    yellow_wgt = np.maximum(yellow_wgt, glow_wgt)
    wgt = cv2.GaussianBlur(wgt, (0, 0), 8)
    # the band is a NARROW strip: the sigma-8 blur above would dilute its
    # weight with the zero-weight green wall next to it (0.9 -> ~0.7 measured),
    # so the yellow term gets its own tighter blur and joins afterwards.
    wgt = np.maximum(wgt, cv2.GaussianBlur(yellow_wgt, (0, 0), 3))
    lab[..., 1] = a*(1-wgt)+128; lab[..., 2] = b*(1-wgt)+128
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _match_white(img, strength=0.85):
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    br = lab[..., 0] > 0.78*255
    if br.sum() < 1000: return img
    lab[..., 1] += (128-float(lab[..., 1][br].mean()))*strength
    lab[..., 2] += (128-float(lab[..., 2][br].mean()))*strength
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _lift_whites(img, amount=0.18, start=0.45, exclude=None):
    """Brighten ONLY the bright zone (walls/ceiling/whites) toward white,
    leaving shadows and midtones untouched. The (1-img) term approaches white
    smoothly instead of clipping; `exclude` (the bright ramp) keeps the
    already-blown window OUT of this lift so only real interior whites move."""
    luma = img.mean(2)
    w = np.clip((luma - start) / (1.0 - start), 0, 1) ** 1.3
    if exclude is not None:
        w = w * (1 - exclude)
    return np.clip(img + amount * w[..., None] * (1.0 - img), 0, 1)


def _scurve(img, s2=0.05):
    return np.clip(img + s2*np.sin(2*np.pi*(img-0.5)), 0, 1)


def _tame_warm(img, knee=16, compress=0.55, l_gain=24):
    """Fix for over-saturated / too-dark BROWNS & ORANGES (terracotta pot,
    dark wood) — measured vs AutoHDR: the pot came out +0.065 more saturated
    and the mahogany woods ~0.05 darker, because the per-channel _scurve
    darkens+saturates warm midtones while everything else gets whiter.

    Warm-quadrant only (LAB a>0 & b>0 soft gate) so the teal chair, sky and
    all cool/neutral colours are untouched:
      1) chroma above `knee` is soft-compressed (tames the pot's saturation)
      2) dark/mid warm pixels get a small L give-back (un-darkens the wood)
    Calibrated on the real photo: pot dL -0.002 / dS +0.009 vs AutoHDR."""
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    A = lab[..., 1]-128; B = lab[..., 2]-128
    ch = np.sqrt(A*A+B*B)
    warm_w = np.clip(A/6, 0, 1) * np.clip(B/10, 0, 1)     # soft brown/orange gate
    over = np.clip(ch - knee, 0, None)
    scale = (knee + over*compress) / np.maximum(ch, 1e-3)
    scale = 1 + (scale - 1) * warm_w
    scale = cv2.GaussianBlur(scale.astype(np.float32), (0, 0), 3)
    lab[..., 1] = A*scale + 128; lab[..., 2] = B*scale + 128
    lw = np.clip((ch-8)/20, 0, 1) * warm_w * np.clip((0.62 - lab[..., 0]/255)/0.35, 0, 1)
    lw = cv2.GaussianBlur(lw.astype(np.float32), (0, 0), 3)
    lab[..., 0] = np.clip(lab[..., 0] + l_gain*lw, 0, 255)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _desat_warm(img, lo=30, hi=115):
    """CLIENT FIX (2026-07): desaturate the YELLOW and ORANGE hues across the
    whole result — the client flagged the warm casts that survive fusion+HDR.

    LAB hue-targeted chroma compression (like Lightroom's HSL yellow/orange
    saturation sliders): hue window 30..115 deg with soft shoulders covers
    red-orange -> yellow; the a-gate hard-protects green walls (a <= -6, sage
    sits at hue ~120 right next to the window) and everything cool. Yellow
    (hue > 70) is compressed harder (60%) than orange (40%) so wood keeps
    some warmth while yellow casts/fabrics go towards neutral."""
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1]-128; b = lab[..., 2]-128
    hue = np.degrees(np.arctan2(b, a))            # orange ~60, yellow ~90
    w = np.clip((hue-lo)/18, 0, 1) * np.clip((hi-hue)/12, 0, 1)
    w *= np.clip(b/6, 0, 1) * np.clip((a+6)/4, 0, 1)
    w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), 3)
    amount = 0.4 + 0.2*np.clip((hue-70)/15, 0, 1)  # orange 0.4 -> yellow 0.6
    f = 1 - amount*w
    lab[..., 1] = a*f + 128; lab[..., 2] = b*f + 128
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _sharpen(img, fine=0.9, clarity=0.2):
    img = np.clip(img + fine*(img - cv2.GaussianBlur(img, (0, 0), 1.0)), 0, 1)
    wide = cv2.GaussianBlur(img, (0, 0), 15)
    # MUDDY FIX 3/3: damp the wide-radius clarity in shadows — full-strength
    # local contrast on already-dark pixels is what turns "dark" into
    # "muddy/gritty". Never fully zero (blacks would go flat/dead).
    luma = img.mean(2)
    damp = np.clip((luma - 0.25) / 0.25, 0.15, 1.0)[..., None]
    return np.clip(img + clarity*damp*(img - wide), 0, 1)
# --------------------------------------------------------------------------------


def process(bracket_imgs, W=2000):
    """Core pipeline, reusable by both the CLI (below) and local_hdr_app.py.

    bracket_imgs : list of BGR uint8 arrays (any size/order) — 2 or more
    returns      : (result BGR uint8, fused BGR uint8) — fused = before finishing
    """
    def _resize(im):
        h, w = im.shape[:2]
        return cv2.resize(im, (W, int(h * W / w)), interpolation=cv2.INTER_AREA) if w != W else im

    bimgs = sorted((_resize(im) for im in bracket_imgs), key=lambda im: im.mean())
    H, Wd = bimgs[0].shape[:2]
    ref = bimgs[len(bimgs) // 2]
    aligned = _align(bimgs, ref, Wd, H)

    fused = (cv2.createMergeMertens().process(aligned) * 255).clip(0, 255).astype("uint8")
    fused = _shadow_fill(fused, aligned[-1])   # crushed shadows <- brightest bracket

    # the CELL 7 chain + muddy fixes + interior-brightness calibration.
    # No segmentation: the "window" is approximated by a pure-luminance bright
    # ramp (hdr.py's proven fallback) so it can be excluded from measurements
    # and shielded from the interior lifts.
    img = fused.astype(np.float32) / 255
    ex = _bright_ramp(img)
    img = _wb(img, exclude=ex)
    img = _levels(img, exclude=ex)
    img = _expose(img, exclude=ex)     # interior median -> 0.62; window shielded
    img = _neutralize(img)             # muddy fix: shadows de-tinted too
    img = _match_white(img)
    img = _lift_whites(img, exclude=ex)  # walls/whites brightened; window shielded
    img = _scurve(img, 0.05)
    img = _tame_warm(img)              # browns/oranges: de-oversaturate + un-darken
    img = _desat_warm(img)             # CLIENT FIX: yellows/oranges desaturated
    img = _sharpen(img)                # muddy fix: clarity damped in shadows

    result = (img * 255).clip(0, 255).astype("uint8")
    return result, fused


def main():
    folder = _ask("folder with the exposure BRACKETS: ")
    bpaths = []
    for e in _EXTS:
        bpaths += glob.glob(os.path.join(folder, e))
    bpaths = sorted(set(os.path.normcase(p) for p in bpaths))
    if len(bpaths) < 2:
        raise SystemExit(f"need >= 2 exposures in {folder!r}, found {len(bpaths)}")

    bimgs = [_load(p) for p in bpaths]
    result, _ = process(bimgs)
    out_path = os.path.join(os.getcwd(), "hdr_result_local.jpg")
    cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"saved -> {out_path}")

    try:
        cv2.imshow("HDR result (press any key to close)", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":
    main()
