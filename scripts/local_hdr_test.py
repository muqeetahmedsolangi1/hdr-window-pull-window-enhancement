"""
LOCAL HDR TEST — pure HDR only: align + Mertens fuse the brackets, then run the
finishing chain from clear_seg_window_pull.py's CELL 7 (Colab), UNCHANGED —
byte-for-byte the same _wb / _levels / _expose(0.70) / _neutralize /
_match_white / _scurve / _sharpen chain. NO segmentation, NO window mask,
NO glass compositing — just the tone-mapping chain on the whole image.

The ONLY difference vs the Colab pipeline: there is no glass mask here, so the
`exclude=` parameter of _wb/_levels is unused (the Colab version excludes the
segmented window glass from those measurements and composites the darker
image's view into the glass afterwards).

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


def _expose(img, target=0.70):
    m = max(float(np.median(img.mean(2))), 1e-3)
    return img if m >= target else np.clip(img, 0, 1) ** float(np.clip(np.log(target)/np.log(m), 0.45, 1.0))


def _neutralize(img):
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]/255; a = lab[..., 1]-128; b = lab[..., 2]-128
    ch = np.sqrt(a*a+b*b)
    wgt = np.clip((L-0.70)/0.12, 0, 1) * np.clip((22-ch)/8, 0, 1)
    wgt = np.maximum(wgt, np.clip((L-0.85)/0.10, 0, 1))
    wgt = cv2.GaussianBlur(wgt, (0, 0), 8)
    lab[..., 1] = a*(1-wgt)+128; lab[..., 2] = b*(1-wgt)+128
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _match_white(img, strength=0.85):
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    br = lab[..., 0] > 0.78*255
    if br.sum() < 1000: return img
    lab[..., 1] += (128-float(lab[..., 1][br].mean()))*strength
    lab[..., 2] += (128-float(lab[..., 2][br].mean()))*strength
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)/255


def _scurve(img, s2=0.05):
    return np.clip(img + s2*np.sin(2*np.pi*(img-0.5)), 0, 1)


def _sharpen(img, fine=0.9, clarity=0.2):
    img = np.clip(img + fine*(img - cv2.GaussianBlur(img, (0, 0), 1.0)), 0, 1)
    wide = cv2.GaussianBlur(img, (0, 0), 15)
    return np.clip(img + clarity*(img - wide), 0, 1)
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

    # the exact CELL 7 chain (no exclude mask here — no segmentation locally)
    img = fused.astype(np.float32) / 255
    img = _wb(img)
    img = _levels(img)
    img = _expose(img, 0.70)
    img = _neutralize(img)
    img = _match_white(img)
    img = _scurve(img, 0.05)
    img = _sharpen(img)

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
