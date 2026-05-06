#!/usr/bin/env python3
"""
Manual bounding box annotation webapp.

Drop images into manual_annotations/images/, then annotate trees with bounding boxes.
Exports CSV compatible with the DeepForest training pipeline.

Keyboard shortcuts:
  ← / →     Prev / Next image
  D          Delete last box
  C          Clear all boxes on current image
  E          Mark image as confirmed empty (no trees) and advance
  Ctrl+S     Export CSV

Run:
  python annotate_manual.py
  Open http://localhost:5051
"""

import json, csv
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, send_file

IMAGES_DIR   = Path("manual_annotations/images")
STATE_FILE   = Path("manual_annotations/annotations.json")
EXPORT_CSV   = Path("manual_annotations/manual_annotations.csv")

SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

app = Flask(__name__)

# ── State ─────────────────────────────────────────────────────────────────────

def get_image_list():
    imgs = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in SUPPORTED)
    return [p.name for p in imgs]


def load_state():
    images = get_image_list()
    if STATE_FILE.exists():
        s = json.loads(STATE_FILE.read_text())
        s.setdefault("confirmed_empty", [])
        # Merge: keep existing annotations, add any new images
        existing = set(s["images"])
        for img in images:
            if img not in existing:
                s["images"].append(img)
                s["annotations"].setdefault(img, [])
        # Remove images that no longer exist
        s["images"] = [i for i in s["images"] if i in set(images)]
        s["confirmed_empty"] = [i for i in s["confirmed_empty"] if i in set(images)]
        return s
    return {
        "images": images,
        "pos": 0,
        "annotations": {img: [] for img in images},
        "confirmed_empty": [],
    }


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def export_csv(state):
    confirmed_empty = set(state.get("confirmed_empty", []))
    rows = []
    for img_name, boxes in state["annotations"].items():
        img_path = str((IMAGES_DIR / img_name).resolve())
        if boxes:
            for box in boxes:
                rows.append({
                    "image_path": img_path,
                    "xmin": round(box["xmin"]),
                    "ymin": round(box["ymin"]),
                    "xmax": round(box["xmax"]),
                    "ymax": round(box["ymax"]),
                    "label": "Tree",
                })
        elif img_name in confirmed_empty:
            # Confirmed hard negative — empty row
            rows.append({
                "image_path": img_path,
                "xmin": "", "ymin": "", "xmax": "", "ymax": "",
                "label": "",
            })
        # Unreviewed images (no boxes, not confirmed empty) are skipped

    with open(EXPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path","xmin","ymin","xmax","ymax","label"])
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["label"] == "Tree")
    n_neg = sum(1 for r in rows if r["label"] == "")
    return {"rows": len(rows), "positives": n_pos, "negatives": n_neg}


# ── Routes ────────────────────────────────────────────────────────────────────

HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Manual Annotation</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #eee; font-family: monospace;
       display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

#topbar { display: flex; align-items: center; gap: 12px; padding: 8px 14px;
          background: #1a1a1a; border-bottom: 1px solid #333; flex-shrink: 0; flex-wrap: wrap; }
#img-label { font-size: 13px; color: #ccc; flex: 1; min-width: 0;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#stats { font-size: 12px; color: #aaa; white-space: nowrap; }
#progress { font-size: 12px; color: #aaa; white-space: nowrap; }

#canvas-wrap { flex: 1; display: flex; align-items: center; justify-content: center;
               overflow: hidden; position: relative; background: #0a0a0a; }
canvas { cursor: crosshair; display: block; }

#botbar { display: flex; gap: 10px; padding: 8px 14px; background: #1a1a1a;
          border-top: 1px solid #333; flex-shrink: 0; align-items: center; flex-wrap: wrap; }
button { padding: 7px 20px; border: none; border-radius: 4px; font-size: 13px;
         cursor: pointer; font-family: monospace; }
#btn-prev   { background: #2a2a2a; color: #eee; }
#btn-next   { background: #2a2a2a; color: #eee; }
#btn-del    { background: #b71c1c; color: #eee; }
#btn-clear  { background: #4a1a1a; color: #eee; }
#btn-empty  { background: #e65100; color: #eee; }
#btn-export { background: #1565c0; color: #eee; }
button:hover { filter: brightness(1.2); }
#hint { font-size: 11px; color: #555; }

#box-list { font-size: 11px; color: #888; flex: 1; min-width: 0;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
</head>
<body>

<div id="topbar">
  <div id="img-label">—</div>
  <div id="stats">0 boxes</div>
  <div id="progress">— / —</div>
</div>

<div id="canvas-wrap">
  <canvas id="canvas"></canvas>
</div>

<div id="botbar">
  <button id="btn-prev"   onclick="navigate(-1)">◀ Prev (←)</button>
  <button id="btn-next"   onclick="navigate(1)">Next (→) ▶</button>
  <button id="btn-del"    onclick="deleteLast()">Del last (D)</button>
  <button id="btn-clear"  onclick="clearAll()">Clear (C)</button>
  <button id="btn-empty"  onclick="markEmpty()">✗ Empty (E)</button>
  <button id="btn-export" onclick="doExport()">💾 Export (⌘S)</button>
  <div id="box-list"></div>
  <div id="hint">click+drag = draw box</div>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

let state     = null;   // server state
let img       = new Image();
let imgNative = {w: 1, h: 1};   // native pixel size
let scale     = 1;               // canvas px per image px
let offsetX   = 0, offsetY = 0; // canvas offset (centred)
let boxes     = [];              // [{xmin,ymin,xmax,ymax}] in image coords
let drawing   = false;
let startX, startY, curX, curY;

// ── Fetch / render ───────────────────────────────────────────────────────────

async function fetchState() {
  const r = await fetch('/state');
  state = await r.json();
  boxes = (state.boxes || []).map(b => ({...b}));
  loadImage();
  updateStats();
}

function loadImage() {
  if (!state || !state.current) {
    document.getElementById('img-label').textContent = 'No images in manual_annotations/images/';
    return;
  }
  document.getElementById('img-label').textContent = state.current;
  document.getElementById('progress').textContent =
    (state.pos + 1) + ' / ' + state.total;

  img = new Image();
  img.onload = () => {
    imgNative.w = img.naturalWidth;
    imgNative.h = img.naturalHeight;
    fitCanvas();
    redraw();
  };
  img.src = '/image/' + encodeURIComponent(state.current) + '?' + Date.now();
  updateStats();
}

function fitCanvas() {
  const wrap = document.getElementById('canvas-wrap');
  const maxW = wrap.clientWidth  - 4;
  const maxH = wrap.clientHeight - 4;
  scale = Math.min(maxW / imgNative.w, maxH / imgNative.h, 1);
  canvas.width  = Math.round(imgNative.w * scale);
  canvas.height = Math.round(imgNative.h * scale);
  offsetX = 0; offsetY = 0;
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  // Draw saved boxes
  boxes.forEach((b, i) => {
    const x = b.xmin * scale, y = b.ymin * scale;
    const w = (b.xmax - b.xmin) * scale, h = (b.ymax - b.ymin) * scale;
    ctx.strokeStyle = '#00e676';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = 'rgba(0,230,118,0.08)';
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = '#00e676';
    ctx.font = '11px monospace';
    ctx.fillText(i + 1, x + 3, y + 12);
  });

  // Draw in-progress box
  if (drawing) {
    const x = Math.min(startX, curX), y = Math.min(startY, curY);
    const w = Math.abs(curX - startX), h = Math.abs(curY - startY);
    ctx.strokeStyle = '#ffeb3b';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }
}

function updateStats() {
  const emptyTag = state && state.confirmed_empty ? ' ✗ empty' : '';
  document.getElementById('stats').textContent =
    boxes.length + ' box' + (boxes.length !== 1 ? 'es' : '') + emptyTag;
  const list = boxes.map((b,i) =>
    `${i+1}:[${Math.round(b.xmin)},${Math.round(b.ymin)},${Math.round(b.xmax)},${Math.round(b.ymax)}]`
  ).join('  ');
  document.getElementById('box-list').textContent = list;
  // Orange border when confirmed empty
  canvas.style.outline = (state && state.confirmed_empty) ? '3px solid #ff6d00' : 'none';
}

// ── Mouse drawing ────────────────────────────────────────────────────────────

function canvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

canvas.addEventListener('mousedown', e => {
  const {x, y} = canvasCoords(e);
  drawing = true;
  startX = x; startY = y; curX = x; curY = y;
});

canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const {x, y} = canvasCoords(e);
  curX = x; curY = y;
  redraw();
});

canvas.addEventListener('mouseup', e => {
  if (!drawing) return;
  drawing = false;
  const {x, y} = canvasCoords(e);
  curX = x; curY = y;

  const x1 = Math.min(startX, curX) / scale;
  const y1 = Math.min(startY, curY) / scale;
  const x2 = Math.max(startX, curX) / scale;
  const y2 = Math.max(startY, curY) / scale;

  if ((x2 - x1) > 4 && (y2 - y1) > 4) {   // ignore tiny accidental drags
    boxes.push({xmin: x1, ymin: y1, xmax: x2, ymax: y2});
    saveBoxes();
  }
  redraw();
  updateStats();
});

// ── Actions ──────────────────────────────────────────────────────────────────

async function saveBoxes() {
  await fetch('/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({boxes})
  });
  updateStats();
}

async function navigate(dir) {
  const r = await fetch('/navigate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({dir})
  });
  state = await r.json();
  boxes = (state.boxes || []).map(b => ({...b}));
  loadImage();
}

function deleteLast() {
  if (boxes.length === 0) return;
  boxes.pop();
  saveBoxes();
  redraw();
  updateStats();
}

function clearAll() {
  boxes = [];
  saveBoxes();
  redraw();
  updateStats();
}

async function markEmpty() {
  const r = await fetch('/mark_empty', {method: 'POST'});
  state = await r.json();
  boxes = (state.boxes || []).map(b => ({...b}));
  loadImage();
}

async function doExport() {
  const r = await fetch('/export', {method: 'POST'});
  const d = await r.json();
  alert(`Exported ${d.rows} rows (${d.positives} tree boxes, ${d.negatives} confirmed-empty images) → manual_annotations/manual_annotations.csv`);
}

// ── Keyboard ─────────────────────────────────────────────────────────────────

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') navigate(1);
  if (e.key === 'ArrowLeft')  navigate(-1);
  if (e.key === 'd' || e.key === 'D') deleteLast();
  if (e.key === 'c' || e.key === 'C') clearAll();
  if (e.key === 'e' || e.key === 'E') markEmpty();
  if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); doExport(); }
});

// ── Resize ───────────────────────────────────────────────────────────────────

window.addEventListener('resize', () => { fitCanvas(); redraw(); });

fetchState();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


def _state_response(s):
    images = s["images"]
    pos = s["pos"]
    if not images:
        return {"current": None, "pos": 0, "total": 0, "boxes": [], "confirmed_empty": False}
    current = images[pos]
    return {
        "current": current,
        "pos": pos,
        "total": len(images),
        "boxes": s["annotations"].get(current, []),
        "confirmed_empty": current in s.get("confirmed_empty", []),
    }


@app.route("/state")
def get_state():
    s = load_state()
    save_state(s)
    return jsonify(_state_response(s))


@app.route("/image/<name>")
def serve_image(name):
    path = IMAGES_DIR / name
    if not path.exists():
        return "not found", 404
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        # Convert to PNG on the fly for browser display
        import io
        import numpy as np
        import rasterio
        from PIL import Image as PILImage
        with rasterio.open(path) as src:
            bands = src.count
            if bands >= 3:
                rgb = src.read([1, 2, 3]).transpose(1, 2, 0).astype(float)
            else:
                gray = src.read(1).astype(float)
                rgb = np.stack([gray, gray, gray], axis=2)
        p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        pil = PILImage.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    mimetype = "image/png" if suffix == ".png" else "image/jpeg"
    return send_file(str(path), mimetype=mimetype)


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json()
    s = load_state()
    current = s["images"][s["pos"]]
    s["annotations"][current] = data["boxes"]
    save_state(s)
    return jsonify({"ok": True})


@app.route("/navigate", methods=["POST"])
def navigate():
    data = request.get_json()
    s = load_state()
    s["pos"] = max(0, min(s["pos"] + data["dir"], len(s["images"]) - 1))
    save_state(s)
    return jsonify(_state_response(s))


@app.route("/mark_empty", methods=["POST"])
def mark_empty():
    s = load_state()
    current = s["images"][s["pos"]]
    ce = s.setdefault("confirmed_empty", [])
    if current in ce:
        # Toggle off
        ce.remove(current)
    else:
        # Toggle on — clear any boxes and mark empty, then advance
        ce.append(current)
        s["annotations"][current] = []
        s["pos"] = min(s["pos"] + 1, len(s["images"]) - 1)
    save_state(s)
    return jsonify(_state_response(s))


@app.route("/export", methods=["POST"])
def export():
    s = load_state()
    result = export_csv(s)
    return jsonify(result)


if __name__ == "__main__":
    print("Drop images into manual_annotations/images/")
    print("Open http://localhost:5051")
    print("Keyboard: ←/→=navigate  D=delete last  C=clear  E=mark empty  Ctrl+S=export")
    app.run(port=5051, debug=False)
