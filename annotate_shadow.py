#!/usr/bin/env python3
"""
Shadow vector annotation web app.
Displays each NEON image; user clicks and drags to define shadow direction.
Saves annotations to shadow_annotations.json, then merges into phase21_train.csv.

Usage:
    source venv310/bin/activate
    python annotate_shadow.py
    # open http://localhost:5000
"""

import json
import math
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_CSV       = Path("phase21/phase21_train.csv")
ANNOTATIONS_OUT = Path("phase21/shadow_annotations.json")
IMAGE_DIR       = Path("manual_annotations/images")

# Load NEON image list from train CSV
_df = pd.read_csv(TRAIN_CSV)
NEON_IMAGES = sorted(
    _df[~_df["image_path"].apply(lambda x: Path(x).name)
        .str.contains("bru|won", case=False)]["image_path"].unique()
)

# Load existing annotations if present
if ANNOTATIONS_OUT.exists():
    annotations = json.loads(ANNOTATIONS_OUT.read_text())
else:
    annotations = {}

app = Flask(__name__)

# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/images")
def list_images():
    return jsonify([
        {"index": i, "path": p, "name": Path(p).name,
         "annotated": Path(p).name in annotations}
        for i, p in enumerate(NEON_IMAGES)
    ])


@app.get("/api/image/<int:idx>")
def serve_image(idx):
    path = Path(NEON_IMAGES[idx])
    return send_file(path, mimetype="image/png")


@app.get("/api/annotations")
def get_annotations():
    return jsonify(annotations)


@app.post("/api/annotations")
def save_annotation():
    data = request.json          # {name, shadow_angle, shadow_x, shadow_y}
    name = data["name"]
    annotations[name] = {
        "shadow_angle": data["shadow_angle"],
        "shadow_x":     data["shadow_x"],
        "shadow_y":     data["shadow_y"],
    }
    ANNOTATIONS_OUT.write_text(json.dumps(annotations, indent=2))
    n_done = len(annotations)
    print(f"  [{n_done}/{len(NEON_IMAGES)}] {name}  angle={data['shadow_angle']:.1f}°")
    return jsonify({"ok": True, "n_done": n_done})


@app.delete("/api/annotations/<name>")
def delete_annotation(name):
    annotations.pop(name, None)
    ANNOTATIONS_OUT.write_text(json.dumps(annotations, indent=2))
    return jsonify({"ok": True})


@app.post("/api/export")
def export_csv():
    """Merge shadow annotations into phase21_train.csv."""
    df = pd.read_csv(TRAIN_CSV)
    df["fname"] = df["image_path"].apply(lambda x: Path(x).name)

    # BRU/WON from phase5
    phase5 = pd.read_csv("deepforest_custom/phase5_train_aug.csv")
    phase5["fname"] = phase5["image_path"].apply(lambda x: Path(x).name)
    p5_lookup = phase5.drop_duplicates("fname").set_index("fname")[
        ["shadow_angle", "shadow_x", "shadow_y", "domain"]
    ]

    def get_shadow(row):
        fname = row["fname"]
        if fname in p5_lookup.index:
            r = p5_lookup.loc[fname]
            return r["shadow_angle"], r["shadow_x"], r["shadow_y"], r["domain"]
        if fname in annotations:
            a = annotations[fname]
            return a["shadow_angle"], a["shadow_x"], a["shadow_y"], "NEON"
        return None, None, None, "NEON"

    df[["shadow_angle", "shadow_x", "shadow_y", "domain"]] = df.apply(
        get_shadow, axis=1, result_type="expand"
    )
    df = df.drop(columns=["fname"])

    out = Path("phase21/phase21_train_with_shadow.csv")
    df.to_csv(out, index=False)
    n_filled = df["shadow_angle"].notna().sum()
    n_total  = len(df)
    return jsonify({"ok": True, "path": str(out),
                    "filled": int(n_filled), "total": int(n_total)})


# ── Frontend ──────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Shadow Vector Annotator</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: monospace; background: #1a1a1a; color: #eee;
       display: flex; flex-direction: column; height: 100vh; }

#toolbar {
  display: flex; align-items: center; gap: 12px;
  padding: 8px 16px; background: #222; border-bottom: 1px solid #444;
  flex-shrink: 0;
}
#toolbar button {
  padding: 6px 14px; border: 1px solid #555; background: #333;
  color: #eee; cursor: pointer; border-radius: 4px; font-size: 13px;
}
#toolbar button:hover { background: #444; }
#toolbar button:disabled { opacity: 0.35; cursor: default; }
#toolbar button.primary { background: #1565c0; border-color: #1976d2; }
#toolbar button.primary:hover { background: #1976d2; }
#toolbar button.danger { background: #7f0000; border-color: #b71c1c; }
#toolbar button.danger:hover { background: #b71c1c; }

#progress { font-size: 13px; color: #aaa; }
#img-name  { font-size: 12px; color: #888; margin-left: auto; max-width: 400px;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#angle-display { font-size: 14px; color: #ffeb3b; min-width: 80px; }
#status { font-size: 12px; color: #4caf50; min-width: 80px; }

#canvas-wrap {
  flex: 1; display: flex; align-items: center; justify-content: center;
  overflow: hidden; position: relative;
}
canvas { cursor: crosshair; max-width: 100%; max-height: 100%; }

#thumbnail-bar {
  display: flex; gap: 4px; padding: 6px 10px; background: #111;
  overflow-x: auto; flex-shrink: 0; border-top: 1px solid #333;
  height: 70px; align-items: center;
}
.thumb {
  width: 52px; height: 52px; object-fit: cover; border: 2px solid #444;
  cursor: pointer; border-radius: 3px; flex-shrink: 0;
  transition: border-color 0.15s;
}
.thumb.active   { border-color: #2196f3; }
.thumb.done     { border-color: #4caf50; }
.thumb.active.done { border-color: #2196f3; }

#instructions {
  position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
  background: rgba(0,0,0,0.65); padding: 5px 12px; border-radius: 4px;
  font-size: 12px; color: #ccc; pointer-events: none;
}
</style>
</head>
<body>

<div id="toolbar">
  <button id="btn-prev" onclick="navigate(-1)">◀ Prev</button>
  <button id="btn-next" onclick="navigate(1)">Next ▶</button>
  <span id="progress">0 / 0</span>
  <button onclick="redo()" class="danger">↺ Redo</button>
  <button id="btn-save" onclick="saveAnnotation()" class="primary" disabled>💾 Save</button>
  <span id="angle-display">—</span>
  <span id="status"></span>
  <button onclick="exportCSV()" style="margin-left:8px">📄 Export CSV</button>
  <span id="img-name"></span>
</div>

<div id="canvas-wrap">
  <canvas id="canvas"></canvas>
  <div id="instructions">Click and drag in the direction shadows point → release to set</div>
</div>

<div id="thumbnail-bar" id="thumbs"></div>

<script>
const canvas  = document.getElementById('canvas');
const ctx     = canvas.getContext('2d');
const thumbBar = document.getElementById('thumbnail-bar');

let images = [];        // [{index, path, name, annotated}]
let annotations = {};   // {name: {shadow_angle, shadow_x, shadow_y}}
let currentIdx = 0;
let img = new Image();

// Drag state
let dragging = false;
let dragStart = null;   // canvas coords
let dragEnd   = null;

// Current drawn vector (normalised, canvas space, for display)
let drawnVec  = null;   // {dx, dy} unit vector in canvas space

// ── Init ─────────────────────────────────────────────────────────────────────

async function init() {
  const [imgResp, annResp] = await Promise.all([
    fetch('/api/images').then(r => r.json()),
    fetch('/api/annotations').then(r => r.json()),
  ]);
  images      = imgResp;
  annotations = annResp;
  buildThumbnails();
  loadImage(0);
}

function buildThumbnails() {
  thumbBar.innerHTML = '';
  images.forEach((im, i) => {
    const el = document.createElement('img');
    el.className = 'thumb' + (im.annotated ? ' done' : '');
    el.src = `/api/image/${i}`;
    el.title = im.name;
    el.onclick = () => loadImage(i);
    el.id = `thumb-${i}`;
    thumbBar.appendChild(el);
  });
}

// ── Image loading ─────────────────────────────────────────────────────────────

function loadImage(idx) {
  currentIdx = idx;
  drawnVec   = null;
  dragStart  = dragEnd = null;

  const im = images[idx];
  document.getElementById('img-name').textContent = im.name;
  document.getElementById('progress').textContent =
    `${Object.keys(annotations).length} / ${images.length} annotated  |  image ${idx+1}/${images.length}`;
  document.getElementById('btn-prev').disabled = idx === 0;
  document.getElementById('btn-next').disabled = idx === images.length - 1;
  document.getElementById('btn-save').disabled = true;
  document.getElementById('status').textContent = annotations[im.name] ? '✓ saved' : '';
  document.getElementById('angle-display').textContent = '—';

  // Highlight active thumb, scroll into view
  document.querySelectorAll('.thumb').forEach(t => t.classList.remove('active'));
  const activeThumb = document.getElementById(`thumb-${idx}`);
  if (activeThumb) { activeThumb.classList.add('active'); activeThumb.scrollIntoView({inline:'center',behavior:'smooth'}); }

  img = new Image();
  img.onload = () => {
    resizeCanvas();
    drawScene();
    // If already annotated, draw saved arrow
    if (annotations[im.name]) {
      const a   = annotations[im.name];
      const cx  = canvas.width  / 2;
      const cy  = canvas.height / 2;
      const len = Math.min(canvas.width, canvas.height) * 0.3;
      drawnVec  = {dx: a.shadow_x, dy: -a.shadow_y};  // flip y for canvas
      dragStart = {x: cx, y: cy};
      dragEnd   = {x: cx + a.shadow_x * len, y: cy - a.shadow_y * len};
      drawScene();
    }
  };
  img.src = `/api/image/${idx}`;
}

function resizeCanvas() {
  const wrap = document.getElementById('canvas-wrap');
  const maxW = wrap.clientWidth  - 20;
  const maxH = wrap.clientHeight - 20;
  const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1);
  canvas.width  = Math.round(img.naturalWidth  * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
}

// ── Drawing ───────────────────────────────────────────────────────────────────

function drawScene() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  if (dragStart && dragEnd) {
    const dx = dragEnd.x - dragStart.x;
    const dy = dragEnd.y - dragStart.y;
    const len = Math.hypot(dx, dy);
    if (len < 5) return;

    // Extend arrow to fixed display length
    const dispLen = Math.min(canvas.width, canvas.height) * 0.35;
    const ex = dragStart.x + (dx / len) * dispLen;
    const ey = dragStart.y + (dy / len) * dispLen;

    // Shadow: slightly offset
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.6)';
    ctx.shadowBlur  = 4;

    // Line
    ctx.beginPath();
    ctx.moveTo(dragStart.x, dragStart.y);
    ctx.lineTo(ex, ey);
    ctx.strokeStyle = '#ffeb3b';
    ctx.lineWidth   = 3;
    ctx.stroke();

    // Arrowhead
    const angle  = Math.atan2(ey - dragStart.y, ex - dragStart.x);
    const hs     = 18;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - hs * Math.cos(angle - 0.4), ey - hs * Math.sin(angle - 0.4));
    ctx.lineTo(ex - hs * Math.cos(angle + 0.4), ey - hs * Math.sin(angle + 0.4));
    ctx.closePath();
    ctx.fillStyle = '#ffeb3b';
    ctx.fill();

    // Origin dot
    ctx.beginPath();
    ctx.arc(dragStart.x, dragStart.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#ffeb3b';
    ctx.fill();
    ctx.restore();

    // Angle label
    const shadowX =  dx / len;
    const shadowY = -dy / len;   // flip canvas y → world y
    const deg = ((Math.atan2(shadowX, shadowY) * 180 / Math.PI) + 360) % 360;
    ctx.fillStyle = '#ffeb3b';
    ctx.font = 'bold 15px monospace';
    ctx.fillText(`${deg.toFixed(1)}°`, ex + 10, ey - 8);

    document.getElementById('angle-display').textContent = `${deg.toFixed(1)}°`;
    document.getElementById('btn-save').disabled = false;

    drawnVec = {dx: shadowX, dy: shadowY, deg};
  }
}

// ── Mouse events ──────────────────────────────────────────────────────────────

canvas.addEventListener('mousedown', e => {
  const r = canvas.getBoundingClientRect();
  dragStart = {x: e.clientX - r.left, y: e.clientY - r.top};
  dragEnd   = {...dragStart};
  dragging  = true;
});

canvas.addEventListener('mousemove', e => {
  if (!dragging) return;
  const r = canvas.getBoundingClientRect();
  dragEnd = {x: e.clientX - r.left, y: e.clientY - r.top};
  drawScene();
});

canvas.addEventListener('mouseup', () => { dragging = false; });
canvas.addEventListener('mouseleave', () => { dragging = false; });

window.addEventListener('resize', () => { if (img.src) { resizeCanvas(); drawScene(); } });

// ── Actions ───────────────────────────────────────────────────────────────────

function navigate(dir) {
  const next = currentIdx + dir;
  if (next >= 0 && next < images.length) loadImage(next);
}

function redo() {
  drawnVec = dragStart = dragEnd = null;
  document.getElementById('btn-save').disabled = true;
  document.getElementById('angle-display').textContent = '—';
  drawScene();
}

async function saveAnnotation() {
  if (!drawnVec) return;
  const im   = images[currentIdx];
  const data = {
    name:         im.name,
    shadow_angle: drawnVec.deg,
    shadow_x:     drawnVec.dx,
    shadow_y:     drawnVec.dy,
  };
  await fetch('/api/annotations', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
  annotations[im.name] = data;
  images[currentIdx].annotated = true;

  // Update thumb
  document.getElementById(`thumb-${currentIdx}`).classList.add('done');
  document.getElementById('status').textContent = '✓ saved';
  document.getElementById('progress').textContent =
    `${Object.keys(annotations).length} / ${images.length} annotated  |  image ${currentIdx+1}/${images.length}`;

  // Auto-advance
  if (currentIdx < images.length - 1) setTimeout(() => navigate(1), 300);
}

async function exportCSV() {
  const resp = await fetch('/api/export', {method:'POST'});
  const data = await resp.json();
  alert(`Exported: ${data.path}\n${data.filled}/${data.total} rows have shadow_angle`);
}

// Keyboard shortcuts
window.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'd') navigate(1);
  if (e.key === 'ArrowLeft'  || e.key === 'a') navigate(-1);
  if (e.key === 'Enter' || e.key === 's') saveAnnotation();
  if (e.key === 'r') redo();
});

init();
</script>
</body>
</html>
"""

@app.get("/")
def index():
    from flask import Response
    return Response(HTML, mimetype="text/html")


if __name__ == "__main__":
    print(f"Shadow annotator — {len(NEON_IMAGES)} NEON images")
    print(f"Existing annotations: {len(annotations)}")
    print("Open http://localhost:5001")
    app.run(debug=False, port=5001)
