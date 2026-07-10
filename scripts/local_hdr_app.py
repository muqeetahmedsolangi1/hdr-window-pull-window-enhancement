"""Minimal web UI for local_hdr_test.py — pure HDR only (no segmentation, no
window mask). Upload the exposure brackets, get back the fused base and the
finishing chain applied to the whole image.

Shows every uploaded bracket as a thumbnail with its EV / shutter / ISO
(sorted darkest -> brightest, the exact order the pipeline uses) and labels
each frame's role — ALL of them are merged by Mertens exposure fusion; the
middle one is the alignment reference.

MANUAL ADJUST: the result card has a "Manual adjust" button that opens a
Lightroom-style modal (dark panel, live preview) with Exposure / Contrast /
Highlights / Shadows / Whites / Blacks / Temp / Tint / Vibrance / Saturation /
Sharpness sliders. Every slider move is applied SERVER-SIDE in numpy on a
cached copy of the result (downscaled preview for speed, debounced), and the
Download button renders the SAME parameters at full resolution.

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

# ------------------------- MANUAL ADJUSTMENTS (server-side) -------------------------
ADJ_KEYS = ("exposure", "contrast", "highlights", "shadows", "whites", "blacks",
            "temp", "tint", "vibrance", "saturation", "sharpness")

# last few processed results kept in memory for the adjust modal:
# rid -> {"full": float32 BGR 0..1, "prev": float32 BGR 0..1 (preview size)}
STATE = {}
_STATE_MAX = 3


def _remember(rid, result_u8, preview_w=1100):
    h, w = result_u8.shape[:2]
    pw = min(preview_w, w)
    prev = cv2.resize(result_u8, (pw, int(h * pw / w)), interpolation=cv2.INTER_AREA)
    STATE[rid] = {"full": result_u8.astype(np.float32) / 255,
                  "prev": prev.astype(np.float32) / 255}
    while len(STATE) > _STATE_MAX:
        STATE.pop(next(iter(STATE)))


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_adjust(img, params):
    """Lightroom-style manual adjustments on a float32 BGR 0..1 image.

    Every slider is -100..+100 (0 = untouched, sliders that are 0 cost
    nothing). Order mirrors the classic develop pipeline: white balance ->
    exposure -> tone regions -> contrast -> vibrance/saturation -> sharpness.
    LAB work uses the FLOAT path (L 0..100, a/b centred on 0) so there is no
    8-bit quantisation between steps.
    """
    g = {}
    for k in ADJ_KEYS:
        try:
            g[k] = float(params.get(k) or 0) / 100.0
        except (TypeError, ValueError):
            g[k] = 0.0
    img = np.clip(np.asarray(img, dtype=np.float32), 0, 1)

    # 1) white balance — temp shifts LAB b (blue<->yellow), tint shifts a
    if g["temp"] or g["tint"]:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 2] += g["temp"] * 18.0
        lab[..., 1] += g["tint"] * 18.0
        img = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0, 1)

    # 2) exposure — multiply in LINEAR light (gamma 2.2), ±1.6 stops
    if g["exposure"]:
        lin = np.power(img, 2.2) * (2.0 ** (g["exposure"] * 1.6))
        img = np.clip(np.power(np.clip(lin, 0, None), 1 / 2.2), 0, 1)

    # 3) tone regions — luminance-keyed smoothstep masks, colour preserved
    #    by scaling all channels with newL/L (no hue shift, no hard clip)
    if g["highlights"] or g["shadows"] or g["whites"] or g["blacks"]:
        L = img.mean(2)
        dL = (g["highlights"] * 0.22 * _smoothstep(L, 0.50, 0.90)
              + g["shadows"] * 0.25 * (1 - _smoothstep(L, 0.12, 0.52))
              + g["whites"] * 0.20 * _smoothstep(L, 0.72, 0.98)
              + g["blacks"] * 0.15 * (1 - _smoothstep(L, 0.02, 0.28)))
        ratio = np.clip(L + dL, 0.001, 1) / np.maximum(L, 1e-4)
        img = np.clip(img * ratio[..., None], 0, 1)

    # 4) contrast — smooth sine s-curve around 0.5 (soft ends, never clips)
    if g["contrast"]:
        img = np.clip(img + g["contrast"] * 0.14 * np.sin(2 * np.pi * (img - 0.5)), 0, 1)

    # 5) vibrance (boosts muted colours more) & saturation (uniform chroma)
    if g["vibrance"] or g["saturation"]:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        a, b = lab[..., 1], lab[..., 2]
        ch = np.sqrt(a * a + b * b)
        f = (1 + 1.0 * g["saturation"]) * \
            (1 + 0.9 * g["vibrance"] * np.clip((40 - ch) / 40, 0, 1))
        f = np.clip(f, 0, 3)
        lab[..., 1] = a * f
        lab[..., 2] = b * f
        img = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0, 1)

    # 6) sharpness — unsharp mask (+) / soften blend (-)
    if g["sharpness"] > 0:
        img = np.clip(img + g["sharpness"] * 1.2 *
                      (img - cv2.GaussianBlur(img, (0, 0), 1.2)), 0, 1)
    elif g["sharpness"] < 0:
        w = -g["sharpness"] * 0.9
        img = np.clip(img * (1 - w) + cv2.GaussianBlur(img, (0, 0), 1.5) * w, 0, 1)
    return img
# -------------------------------------------------------------------------------------

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

 /* ---------- manual-adjust modal (Lightroom-style dark) ---------- */
 #adjmodal{position:fixed;inset:0;background:#0d0d0d;display:none;z-index:50}
 #adjmodal.open{display:flex}
 #adjview{flex:1;display:flex;align-items:center;justify-content:center;min-width:0;padding:14px}
 #adjimg{max-width:100%;max-height:calc(100vh - 28px);border-radius:4px;border:1px solid #222;
         user-select:none;-webkit-user-drag:none;cursor:pointer}
 #adjpanel{width:272px;flex:0 0 272px;background:#1b1b1b;color:#ccc;padding:14px 16px;
           overflow-y:auto;border-left:1px solid #2a2a2a;display:flex;flex-direction:column}
 #adjpanel h3{margin:0 0 4px;font-size:14px;color:#eee;font-weight:600}
 #adjpanel .hint{font-size:11px;color:#777;margin-bottom:10px}
 .srow{margin:7px 0}
 .srow .top{display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:2px}
 .srow .top span.val{color:#9ad;min-width:28px;text-align:right}
 .srow label{font-weight:400;font-size:12.5px;color:#ccc;cursor:pointer}
 .srow input[type=range]{width:100%;accent-color:#4a90d9;height:20px;background:transparent;cursor:pointer}
 .abtns{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
 .abtns button{flex:1;padding:8px 6px;font-size:13px;border-radius:6px}
 #adjreset{background:#333}
 #adjclose{background:#444}
 #adjdl{background:#0b6}
 #adjbusy{font-size:11px;color:#888;height:14px;margin-top:8px;text-align:center}
 #cmphint{font-size:10.5px;color:#666;text-align:center;margin-top:4px}
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
  &nbsp; <button type="button" onclick="openAdj()" style="padding:6px 14px;font-size:14px">&#127899; Manual adjust</button>
  <div class="grid">
    <div class="thumb"><b>Fused (Mertens, before finishing)</b><img src="/result/{{ fused }}"></div>
    <div class="thumb result"><b>HDR result</b><br><a href="/result/{{ result }}" download>download</a>
      <img src="/result/{{ result }}"></div>
  </div>
</div>

<div id="adjmodal">
  <div id="adjview">
    <img id="adjimg" title="hold the mouse down to see BEFORE">
  </div>
  <div id="adjpanel">
    <h3>Manual adjust</h3>
    <div class="hint">live preview &middot; double-click a name to reset that slider</div>
    <div id="sliders"></div>
    <div id="adjbusy"></div>
    <div class="abtns">
      <button type="button" id="adjreset" onclick="resetAdj()">Reset</button>
      <button type="button" id="adjdl" onclick="dlAdj()">Download</button>
      <button type="button" id="adjclose" onclick="closeAdj()">Close</button>
    </div>
    <div id="cmphint">hold mouse on the image = before</div>
  </div>
</div>

<script>
const RID = "{{ rid }}";
const SLIDERS = [
 ["exposure","Exposure"],["contrast","Contrast"],["highlights","Highlights"],
 ["shadows","Shadows"],["whites","Whites"],["blacks","Blacks"],
 ["temp","Temp"],["tint","Tint"],["vibrance","Vibrance"],
 ["saturation","Saturation"],["sharpness","Sharpness"]];
const P = {};
SLIDERS.forEach(([k]) => P[k] = 0);

let debT = null, busy = false, pending = false, baseURL = null, curURL = null;

function buildSliders(){
  const host = document.getElementById('sliders');
  host.innerHTML = '';
  SLIDERS.forEach(([key, name]) => {
    const row = document.createElement('div'); row.className = 'srow';
    row.innerHTML =
      `<div class="top"><label id="lb_${key}">${name}</label><span class="val" id="v_${key}">0</span></div>` +
      `<input type="range" id="s_${key}" min="-100" max="100" step="1" value="0">`;
    host.appendChild(row);
    const inp = row.querySelector('input');
    inp.addEventListener('input', () => {
      P[key] = parseInt(inp.value, 10) || 0;
      document.getElementById('v_' + key).textContent = P[key];
      sched();
    });
    row.querySelector('label').addEventListener('dblclick', () => {
      P[key] = 0; inp.value = 0;
      document.getElementById('v_' + key).textContent = 0;
      sched();
    });
  });
}

function sched(){ clearTimeout(debT); debT = setTimeout(update, 130); }

async function update(){
  if (busy){ pending = true; return; }
  busy = true;
  document.getElementById('adjbusy').textContent = 'rendering…';
  try {
    const res = await fetch('/adjust/' + RID, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(P)
    });
    if (res.status === 410){
      alert('This result expired (app was restarted) — process the brackets again.');
      busy = false; return;
    }
    const url = URL.createObjectURL(await res.blob());
    const im = document.getElementById('adjimg');
    im.src = url;
    if (curURL && curURL !== baseURL) URL.revokeObjectURL(curURL);
    curURL = url;
    if (baseURL === null) baseURL = url;   // the all-zero render = BEFORE
  } catch(e){ /* server gone — leave last preview */ }
  document.getElementById('adjbusy').textContent = '';
  busy = false;
  if (pending){ pending = false; update(); }
}

function openAdj(){
  document.getElementById('adjmodal').classList.add('open');
  if (!document.getElementById('sliders').childElementCount) buildSliders();
  if (!curURL) update();          // first open: render the untouched result
}
function closeAdj(){ document.getElementById('adjmodal').classList.remove('open'); }
function resetAdj(){
  SLIDERS.forEach(([k]) => {
    P[k] = 0;
    document.getElementById('s_' + k).value = 0;
    document.getElementById('v_' + k).textContent = 0;
  });
  sched();
}
function dlAdj(){
  const q = new URLSearchParams(P).toString();
  window.location = '/adjust-full/' + RID + '?' + q;
}

// hold-to-compare: while the mouse is down on the image, show the BEFORE
(function(){
  const im = document.getElementById('adjimg');
  const showBase = e => { if (baseURL){ im.src = baseURL; } e.preventDefault(); };
  const showCur  = () => { if (curURL){ im.src = curURL; } };
  im.addEventListener('mousedown', showBase);
  im.addEventListener('touchstart', showBase);
  ['mouseup','mouseleave','touchend'].forEach(ev => im.addEventListener(ev, showCur));
})();
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeAdj(); });
</script>
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


@app.route("/adjust/<rid>", methods=["POST"])
def adjust_preview(rid):
    st = STATE.get(rid)
    if st is None:
        return ("expired", 410)
    params = request.get_json(force=True, silent=True) or {}
    out = apply_adjust(st["prev"], params)
    ok, buf = cv2.imencode(".jpg", (out * 255).clip(0, 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, 88])
    return app.response_class(buf.tobytes(), mimetype="image/jpeg")


@app.route("/adjust-full/<rid>")
def adjust_full(rid):
    st = STATE.get(rid)
    if st is None:
        return ("expired — process the brackets again", 410)
    params = {k: request.args.get(k, 0, type=float) for k in ADJ_KEYS}
    out = apply_adjust(st["full"], params)
    ok, buf = cv2.imencode(".jpg", (out * 255).clip(0, 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    resp = app.response_class(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Content-Disposition"] = f"attachment; filename=hdr_adjusted_{rid}.jpg"
    return resp


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
    rid = str(stamp)
    result_name = f"local_hdr_{stamp}.jpg"
    fused_name = f"local_fused_{stamp}.jpg"
    cv2.imwrite(os.path.join(RESULTS, result_name), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(RESULTS, fused_name), fused, [cv2.IMWRITE_JPEG_QUALITY, 90])
    _remember(rid, result)             # cache for the manual-adjust modal

    return render_template_string(PAGE, result=result_name, fused=fused_name,
                                  brackets=brackets, rid=rid,
                                  took=round(time.time() - t0, 1))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=False)
