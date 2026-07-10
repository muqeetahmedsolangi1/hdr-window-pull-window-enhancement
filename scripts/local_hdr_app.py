"""Minimal web UI for local_hdr_test.py — pure HDR only (no segmentation, no
window mask). Upload the exposure brackets, get back the fused base and the
"old" CELL 7 finishing chain applied to the whole image.

Run:
    python local_hdr_app.py
then open http://127.0.0.1:5003
"""
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import cv2
import numpy as np
from flask import Flask, request, render_template_string, send_from_directory

from local_hdr_test import process

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
 .thumb{border:1px solid #e3e3e3;border-radius:8px;padding:8px}
 .result img{border:1px solid #ddd;margin-top:8px}
 input[type=range]{width:100%;max-width:400px;vertical-align:middle}
 .rangeval{font-weight:700;color:#0a5;margin-left:8px}
</style></head><body>
<h1>local HDR test &mdash; pure HDR, no segmentation</h1>
<div class="sub">Align + Mertens fuse, then the unchanged _wb / _levels / _expose /
_neutralize / _match_white / _scurve / _sharpen chain (same as Colab CELL 7) on
the WHOLE image &mdash; no window mask, no glass compositing.</div>
<form class="card" method="post" enctype="multipart/form-data">
  <div class="field">
    <label>Exposure BRACKETS (2 or more)</label>
    <input type="file" name="brackets" accept="image/*" multiple required>
  </div>
  <button type="submit">Process</button>
</form>
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


def _read(f):
    raw = f.read()
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


@app.route("/result/<path:name>")
def result_file(name):
    return send_from_directory(RESULTS, name)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE)

    bfiles = request.files.getlist("brackets")
    bimgs = [im for im in (_read(f) for f in bfiles) if im is not None]
    if len(bimgs) < 2:
        return render_template_string(PAGE, error="Upload at least 2 exposure brackets.")

    t0 = time.time()
    result, fused = process(bimgs)
    stamp = int(time.time() * 1000)
    result_name = f"local_hdr_{stamp}.jpg"
    fused_name = f"local_fused_{stamp}.jpg"
    cv2.imwrite(os.path.join(RESULTS, result_name), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(RESULTS, fused_name), fused, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return render_template_string(PAGE, result=result_name, fused=fused_name,
                                  took=round(time.time() - t0, 1))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=False)
