"""CLI runner for hdr-enhance.

Usage:
    python run.py <exposures_folder> [output.jpg] [max_width] [no-enhance]

Examples:
    python run.py "D:/new-exposure/test5-exposure" hdr.jpg 4000
    python run.py "D:/new-exposure/test5-exposure" raw.jpg 4000 no-enhance
"""
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import cv2
from hdr import load_exposures, process_brackets


def main():
    if len(sys.argv) < 2:
        print("usage: python run.py <exposures_folder> [output.jpg] [max_width] [no-enhance] [no-pull]\n"
              "  no-enhance = plain Mertens fusion + sharpen only (no finishing chain)\n"
              "  no-pull    = skip the window pull (leave blown windows as-is)")
        return
    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "hdr_result.jpg"
    max_width = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    enhance = "no-enhance" not in sys.argv[1:]
    pull = "no-pull" not in sys.argv[1:]

    imgs = load_exposures(folder)
    print(f"loaded {len(imgs)} exposures from {folder}  (enhance={enhance}, pull_windows={pull})")
    t0 = time.time()
    res = process_brackets(imgs, enhance=enhance, max_width=max_width, pull_windows=pull)
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    if not cv2.imwrite(out, res, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise IOError(f"failed to write {out}")
    print(f"done in {time.time() - t0:.1f}s  ->  {out}  ({res.shape[1]}x{res.shape[0]})")


if __name__ == "__main__":
    main()
