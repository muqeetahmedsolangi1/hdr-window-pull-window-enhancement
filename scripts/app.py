"""Minimal web UI for hdr-enhance.

Upload bracketed exposures of a room; get back ONE merged + finished HDR image
(the pixelswift "HDR enhance" checkbox), pure OpenCV.

Run:
    python app.py
then open http://127.0.0.1:5002
"""
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import cv2
import numpy as np
from flask import Flask, request, render_template_string, send_from_directory

from hdr import process_brackets

app = Flask(__name__)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RESULTS = os.path.abspath(RESULTS)
os.makedirs(RESULTS, exist_ok=True)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>hdr-enhance</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:30px auto;padding:0 16px;color:#222}
 h1{font-size:22px} .sub{color:#666;font-size:14px;margin-bottom:18px}
 .card{border:1px solid #e3e3e3;border-radius:10px;padding:18px;margin:14px 0}
 input[type=file]{margin:8px 0} label{font-size:14px}
 button{background:#0b6;color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:15px;cursor:pointer}
 img{max-width:100%;border-radius:8px;margin-top:12px;border:1px solid #ddd}
 .opt{font-size:13px;color:#555;margin:6px 0}
</style></head><body>
<h1>hdr-enhance &mdash; real-estate HDR (Mertens + white-HDR finishing)</h1>
<div class="sub">Upload the bracketed exposures (2+). Classic Mertens exposure fusion
followed by the calibrated &ldquo;white HDR&rdquo; finishing chain &mdash; pure OpenCV, no AI.</div>
<form class="card" method="post" enctype="multipart/form-data">
  <input type="file" name="brackets" accept="image/*" multiple required><br>
  <div class="opt">
    <label>Max width <input type="number" name="max_width" value="3000" style="width:80px"></label>
  </div>
  <div class="opt">
    <label><input type="checkbox" name="enhance" checked> HDR enhance (calibrated color/tone finishing)</label>
  </div>
  <p><button type="submit">Process</button></p>
</form>
{% if result %}
<div class="card"><b>Result</b> &mdash; {{ took }}s
  <br><a href="/result/{{ result }}" download>download</a>
  <br><img src="/result/{{ result }}"></div>
{% endif %}
{% if error %}<div class="card" style="color:#b00">{{ error }}</div>{% endif %}
</body></html>"""


@app.route("/result/<path:name>")
def result(name):
    return send_from_directory(RESULTS, name)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE)
    files = request.files.getlist("brackets")
    imgs = []
    for f in files:
        data = np.frombuffer(f.read(), np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if im is not None:
            imgs.append(im)
    if len(imgs) < 2:
        return render_template_string(PAGE, error="Upload at least 2 exposure images.")
    try:
        mw = int(request.form.get("max_width", 3000))
    except ValueError:
        mw = 3000
    t0 = time.time()
    res = process_brackets(imgs, enhance=("enhance" in request.form), max_width=mw)
    name = f"hdr_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(os.path.join(RESULTS, name), res, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return render_template_string(PAGE, result=name, took=round(time.time() - t0, 1))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
