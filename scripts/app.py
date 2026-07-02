"""Minimal web UI for hdr-enhance.

Upload bracketed exposures of a room; get back ONE merged + finished HDR image
(Mertens fusion + white-HDR finishing + window pull), pure OpenCV. The page also
shows every uploaded bracket as a thumbnail with its EV / shutter / ISO, sorted the
way the pipeline uses them (darkest -> brightest), and labels which frames have a
special role (all are merged by Mertens; the darkest is the window-pull source and
the middle one is the alignment reference).

Run:
    python app.py
then open http://127.0.0.1:5002
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

from hdr import process_brackets

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
<title>hdr-enhance</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:30px auto;padding:0 16px;color:#222}
 h1{font-size:22px} .sub{color:#666;font-size:14px;margin-bottom:18px}
 .card{border:1px solid #e3e3e3;border-radius:10px;padding:18px;margin:14px 0}
 input[type=file]{margin:8px 0} label{font-size:14px}
 button{background:#0b6;color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:15px;cursor:pointer}
 img{max-width:100%;border-radius:8px}
 .opt{font-size:13px;color:#555;margin:6px 0}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-top:12px}
 .thumb{border:1px solid #e3e3e3;border-radius:8px;padding:6px;font-size:11.5px;line-height:1.35}
 .thumb img{border:1px solid #ddd;display:block}
 .ev{font-weight:700;color:#0a5}
 .fn{color:#888;word-break:break-all}
 .role{display:inline-block;margin-top:3px;padding:1px 6px;border-radius:6px;background:#eef6ff;color:#0366d6;font-weight:600}
 .role.pull{background:#fdeef7;color:#b0367a}
 .result img{border:1px solid #ddd;margin-top:12px}
</style></head><body>
<h1>hdr-enhance &mdash; real-estate HDR (Mertens fusion + white-HDR finishing)</h1>
<div class="sub">Upload the bracketed exposures (2+). Mertens exposure fusion + the
calibrated &ldquo;white HDR&rdquo; finishing on the INDOOR; the window keeps its crisp
fused view (the clearest bracket) and is never blown out &mdash; pure OpenCV.</div>
<form class="card" method="post" enctype="multipart/form-data">
  <input type="file" name="brackets" accept="image/*" multiple required><br>
  <div class="opt">
    <label>Max width <input type="number" name="max_width" value="3000" style="width:80px"></label>
  </div>
  <div class="opt">
    <label><input type="checkbox" name="enhance" checked> HDR enhance (indoor tone/colour; the window keeps its crisp fused view)</label>
  </div>
  <p><button type="submit">Process</button></p>
</form>
{% if brackets %}
<div class="card">
  <b>Merged frames &mdash; {{ brackets|length }} brackets</b>
  <div class="sub" style="margin:4px 0 0">Every frame below is merged by Mertens exposure fusion
  (shown darkest&nbsp;&rarr;&nbsp;brightest, the order the pipeline uses).</div>
  <div class="grid">
  {% for b in brackets %}
    <div class="thumb">
      <img src="{{ b.thumb }}">
      <div><span class="ev">{{ b.ev }}</span> &middot; {{ b.shutter }}</div>
      <div>f/{{ b.fnum }} &middot; ISO {{ b.iso }}</div>
      <div class="fn">{{ b.name }}</div>
      {% if b.win_src %}<span class="role pull">★ window view source</span>{% endif %}
      {% if b.role %}<span class="role {{ b.cls }}">{{ b.role }}</span>{% endif %}
    </div>
  {% endfor %}
  </div>
</div>
{% endif %}
{% if result %}
<div class="card result"><b>Result</b> &mdash; {{ took }}s &middot; enhance={{ enhance }}
  <br><a href="/result/{{ result }}" download>download</a>
  <br><img src="/result/{{ result }}"></div>
{% endif %}
{% if debug %}
<div class="card">
  <b>Debug &mdash; how the window is separated</b>
  <div class="sub" style="margin:4px 0 0">The scene is split into parts, each segmented on its own:
  the <b>surround</b> (curtains/blinds/valances) is deducted and gets the normal indoor HDR;
  only the <b>outdoor view</b> (glass) receives the crisp exterior; the <b>frame</b> comes from the
  lit frame bracket.</div>
  <div class="grid">
    {% if debug.segments %}<div class="thumb"><img src="/result/{{ debug.segments }}">
      <div><b>All segments</b> &mdash; <span style="color:#08a;font-weight:700">cyan</span> glass/view &middot; <span style="color:#e70;font-weight:700">orange</span> frame &middot; <span style="color:#c0c;font-weight:700">magenta</span> curtains</div></div>{% endif %}
    {% if debug.fused %}<div class="thumb"><img src="/result/{{ debug.fused }}">
      <div><b>Fused (Mertens)</b> &mdash; the base going in</div></div>{% endif %}
    {% if debug.surround %}<div class="thumb"><img src="/result/{{ debug.surround }}">
      <div><span style="color:#c0c;font-weight:700">magenta</span> = surround (curtains / blinds) &mdash; deducted, gets indoor HDR</div></div>{% endif %}
    {% if debug.window_excluded %}<div class="thumb"><img src="/result/{{ debug.window_excluded }}">
      <div><span style="color:#0a0;font-weight:700">green</span> = whole window unit kept OUT of the indoor enhance</div></div>{% endif %}
    {% if debug.window_glass %}<div class="thumb"><img src="/result/{{ debug.window_glass }}">
      <div><span style="color:#08a;font-weight:700">cyan</span> = OUTDOOR VIEW (glass) &mdash; crisp exterior composited here</div></div>{% endif %}
    {% if debug.window_frame %}<div class="thumb"><img src="/result/{{ debug.window_frame }}">
      <div><span style="color:#e70;font-weight:700">orange</span> = FRAME / mullions / rail &mdash; from the lit frame bracket</div></div>{% endif %}
    {% if debug.outdoor %}<div class="thumb"><img src="/result/{{ debug.outdoor }}">
      <div><b>outdoor view</b> (approx) &mdash; <span style="color:#08a">sky</span> / <span style="color:#0aa">water</span> / <span style="color:#0a0">green</span> / <span style="color:#e70">other</span></div></div>{% endif %}
  </div>
  <div class="sub" style="margin:8px 0 0">Boundaries are refined with <b>MobileSAM</b> (pixel-precise, edge-snapped); the outdoor breakdown is best-effort.</div>
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


def _build_brackets(items, window_bracket=None):
    """items: list of dicts with keys img, exptime, fnum, iso, name (already sorted
    darkest -> brightest). Adds EV (relative stops vs the median exposure time), a
    thumbnail, and the pipeline role of each frame. `window_bracket` is the index of
    the bracket the pipeline chose for the crisp window view."""
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
            role, cls = "darkest", ""
        elif i == mid_idx:
            role, cls = "middle · alignment reference", ""
        else:
            role, cls = "", ""
        win_src = (window_bracket is not None and i == window_bracket)
        out.append({
            "thumb": _thumb_uri(it["img"]),
            "name": it["name"],
            "ev": ev_str,
            "shutter": _shutter_str(t),
            "fnum": it["fnum"] if it["fnum"] else "&ndash;",
            "iso": it["iso"] if it["iso"] else "&ndash;",
            "role": role,
            "cls": cls,
            "win_src": win_src,
        })
    return out


@app.route("/result/<path:name>")
def result(name):
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
        return render_template_string(PAGE, error="Upload at least 2 exposure images.")

    # darkest -> brightest, exactly the order process_brackets uses internally
    items.sort(key=lambda it: it["mean"])

    try:
        mw = int(request.form.get("max_width", 3000))
    except ValueError:
        mw = 3000
    enhance = "enhance" in request.form

    t0 = time.time()
    res, dbg = process_brackets([it["img"] for it in items], enhance=enhance,
                                max_width=mw, pull_windows=False, return_debug=True)
    # mark, in the gallery, which bracket the pipeline chose for the crisp window view
    brackets = _build_brackets(items, window_bracket=dbg.get("window_bracket"))
    stamp = int(time.time() * 1000)
    name = f"hdr_{stamp}.jpg"
    cv2.imwrite(os.path.join(RESULTS, name), res, [cv2.IMWRITE_JPEG_QUALITY, 95])
    debug = {}
    for key in ("segments", "fused", "surround", "window_excluded", "window_glass",
                "window_frame", "outdoor"):
        if dbg.get(key) is not None:
            dn = f"dbg_{key}_{stamp}.jpg"
            cv2.imwrite(os.path.join(RESULTS, dn), dbg[key], [cv2.IMWRITE_JPEG_QUALITY, 88])
            debug[key] = dn
    return render_template_string(PAGE, result=name, took=round(time.time() - t0, 1),
                                  brackets=brackets, enhance=enhance, debug=debug)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
