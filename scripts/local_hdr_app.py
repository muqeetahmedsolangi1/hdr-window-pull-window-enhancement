"""Minimal web UI for local_hdr_test.py — pure HDR only (no segmentation, no
window mask). Upload the exposure brackets, get back the fused base and the
finishing chain applied to the whole image.

Shows every uploaded bracket as a thumbnail with its EV / shutter / ISO
(sorted darkest -> brightest, the exact order the pipeline uses) and labels
each frame's role — ALL of them are merged by Mertens exposure fusion; the
middle one is the alignment reference.

Run:
    python local_hdr_app.py
then open http://127.0.0.1:5003
"""
import os
import io
import time
import math
import base64

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import cv2
import numpy as np
from flask import Flask, request, render_template_string, send_from_directory

from local_hdr_test import process

try:                                  # EXIF (exposure) reading is optional
    from PIL import Image, ExifTags
    _EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
except Exception:                     # pragma: no cover
    Image = None
    _EXIF_TAGS = {}

app = Flask(__name__)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RESULTS = os.path.abspath(RESULTS)
os.makedirs(RESULTS, exist_ok=True)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>local HDR test</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:30px auto;padding:0 16px;color:#222}
 h1{font-size:22px} .sub{color:#666;font-size:14px;margin-bottom:18px}
 .card{border:1px solid #e3e3e3;border-radius:10px;padding:18px;margin:14px 0}
 input[type=file]{margin:8px 0;display:block} label{font-size:14px;font-weight:600}
 button{background:#0b6;color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:15px;cursor:pointer}
 img{max-width:100%;border-radius:8px}
 .field{margin:14px 0}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px}
 .bgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-top:12px}
 .thumb{border:1px solid #e3e3e3;border-radius:8px;padding:6px;font-size:11.5px;line-height:1.35}
 .thumb img{border:1px solid #ddd;display:block}
 .ev{font-weight:700;color:#0a5}
 .fn{color:#888;word-break:break-all}
 .role{display:inline-block;margin-top:3px;padding:1px 6px;border-radius:6px;background:#eef6ff;color:#0366d6;font-weight:600}
 .result img{border:1px solid #ddd;margin-top:8px}
</style></head><body>
<h1>local HDR test &mdash; pure HDR, no segmentation</h1>
<div class="sub">Align + Mertens fuse, then the finishing chain (wb / levels / expose /
neutralize / match-white / lift-whites / s-curve / tame-warm / sharpen) on the WHOLE
image &mdash; no window mask, no glass compositing.</div>
<form class="card" method="post" enctype="multipart/form-data">
  <div class="field">
    <label>Exposure BRACKETS (2 or more)</label>
    <input type="file" name="brackets" accept="image/*" multiple required>
  </div>
  <button type="submit">Process</button>
</form>
{% if brackets %}
<div class="card">
  <b>Merged frames &mdash; ALL {{ brackets|length }} brackets go into the Mertens fusion</b>
  <div class="sub" style="margin:4px 0 0">Shown darkest &rarr; brightest, the exact order the
  pipeline uses. Mertens weighs every pixel of every frame by contrast / saturation /
  well-exposedness and blends them &mdash; no frame is skipped.</div>
  <div class="bgrid">
  {% for b in brackets %}
    <div class="thumb">
      <img src="{{ b.thumb }}">
      <div><span class="ev">{{ b.ev }}</span> &middot; {{ b.shutter }}</div>
      <div>f/{{ b.fnum }} &middot; ISO {{ b.iso }}</div>
      <div class="fn">{{ b.name }}</div>
      {% if b.role %}<span class="role">{{ b.role }}</span>{% endif %}
    </div>
  {% endfor %}
  </div>
</div>
{% endif %}
{% if result %}
<div class="card">
  <b>Result</b> &mdash; {{ took }}s
  <div class="grid">
    <div class="thumb"><b>Fused (Mertens, before finishing)</b><img src="/result/{{ fused }}"></div>
    <div class="thumb result"><b>HDR result</b><br><a href="/result/{{ result }}" download>download</a>
      <img src="/result/{{ result }}"></div>
  </div>
</div>
{% endif %}
{% if error %}<div class="card" style="color:#b00">{{ error }}</div>{% endif %}
</body></html>"""


def _exif_exposure(raw):
    """Return {exptime, fnum, iso} from a JPEG's EXIF, any value may be None."""
    out = {"exptime": None, "fnum": None, "iso": None}
    if Image is None:
        return out
    try:
        ex = Image.open(io.BytesIO(raw))._getexif() or {}
    except Exception:
        return out

    def g(name):
        return ex.get(_EXIF_TAGS.get(name))
    try:
        out["exptime"] = float(g("ExposureTime")) if g("ExposureTime") else None
    except Exception:
        pass
    try:
        out["fnum"] = round(float(g("FNumber")), 1) if g("FNumber") else None
    except Exception:
        pass
    iso = g("ISOSpeedRatings")
    if isinstance(iso, (list, tuple)):
        iso = iso[0] if iso else None
    out["iso"] = iso
    return out


def _shutter_str(t):
    if not t:
        return "&ndash;"
    return f"1/{round(1.0 / t)}s" if t < 1 else f"{t:g}s"


def _thumb_uri(im, width=220):
    h, w = im.shape[:2]
    th = max(1, int(h * width / w))
    small = cv2.resize(im, (width, th), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _build_brackets(items):
    """items: list of dicts (img, exptime, fnum, iso, name), sorted darkest ->
    brightest. Adds EV (relative stops vs the median exposure time), thumbnail
    and the pipeline role of each frame."""
    times = [it["exptime"] for it in items if it["exptime"]]
    med = sorted(times)[len(times) // 2] if times else None
    mid_idx = len(items) // 2
    out = []
    for i, it in enumerate(items):
        t = it["exptime"]
        if t and med:
            ev = round(math.log2(t / med), 1)
            ev_str = f"EV {ev:+g}" if ev else "EV 0"
        else:
            ev_str = "EV &ndash;"
        if i == 0:
            role = "darkest"
        elif i == mid_idx:
            role = "middle · alignment reference"
        elif i == len(items) - 1:
            role = "brightest"
        else:
            role = ""
        out.append({
            "thumb": _thumb_uri(it["img"]),
            "name": it["name"],
            "ev": ev_str,
            "shutter": _shutter_str(t),
            "fnum": it["fnum"] if it["fnum"] else "&ndash;",
            "iso": it["iso"] if it["iso"] else "&ndash;",
            "role": role,
        })
    return out


@app.route("/result/<path:name>")
def result_file(name):
    return send_from_directory(RESULTS, name)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE)

    items = []
    for f in request.files.getlist("brackets"):
        raw = f.read()
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        ex = _exif_exposure(raw)
        items.append({"img": im, "name": f.filename or "image",
                      "mean": float(im.mean()), **ex})
    if len(items) < 2:
        return render_template_string(PAGE, error="Upload at least 2 exposure brackets.")

    # darkest -> brightest, exactly the order process() uses internally
    items.sort(key=lambda it: it["mean"])
    brackets = _build_brackets(items)

    t0 = time.time()
    result, fused = process([it["img"] for it in items])
    stamp = int(time.time() * 1000)
    result_name = f"local_hdr_{stamp}.jpg"
    fused_name = f"local_fused_{stamp}.jpg"
    cv2.imwrite(os.path.join(RESULTS, result_name), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(RESULTS, fused_name), fused, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return render_template_string(PAGE, result=result_name, fused=fused_name,
                                  brackets=brackets, took=round(time.time() - t0, 1))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=False)
