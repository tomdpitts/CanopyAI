#!/usr/bin/env python3
"""
review_tcd_shadows.py — Web app for reviewing and correcting TCD shadow vectors.

Shows each TCD tile with the auto-estimated shadow direction overlaid as an
arrow. Low-confidence tiles are shown first. Click anywhere on the image to
override the direction (arrow points from centre to click point). Corrections
are written back to data/tcd/tcd_shadow_vectors.json immediately.

Usage:
    source venv310/bin/activate
    python deepforest_custom/tcd_shadow/review_tcd_shadows.py
    # open http://localhost:5055

    # Start on low-confidence tiles only (consensus < 60%):
    python deepforest_custom/tcd_shadow/review_tcd_shadows.py --min-consensus 60

Keyboard shortcuts:
    A     Accept auto-estimate (marks as reviewed, moves to next)
    S     Skip (move to next without saving)
    ←/→   Previous / Next tile
"""

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deepforest_custom"))

DEFAULT_JSON    = ROOT / "data/tcd/tcd_shadow_vectors.json"
DEFAULT_TCD_DIR = ROOT / "data/tcd/images/data/tcd/raw"
THUMBNAIL_SIZE  = 900   # px — displayed size in browser


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",          default=str(DEFAULT_JSON))
    ap.add_argument("--tcd-dir",       default=str(DEFAULT_TCD_DIR))
    ap.add_argument("--min-consensus", type=float, default=60.0,
                    help="Tiles below this consensus are flagged (default 60)")
    ap.add_argument("--filter",        default="low",
                    choices=["low", "high", "all", "unreviewed", "reviewed"],
                    help="Which tiles to show: low=low-confidence, high=high-confidence not yet reviewed, "
                         "all, unreviewed=not manually reviewed, reviewed=already manually reviewed")
    ap.add_argument("--port",          type=int, default=5055)
    ap.add_argument("--max-tiles",     type=int, default=1000,
                    help="Max tiles to load (default 1000)")
    ap.add_argument("--sort",          default="asc", choices=["asc", "desc"],
                    help="asc=low confidence first, desc=high confidence first (default asc)")
    return ap.parse_args()


args      = parse_args()
json_path = Path(args.json)
tcd_dir   = Path(args.tcd_dir)
app       = Flask(__name__)

# ── State — loaded once at startup, updated in-memory after saves ─────────────

_vectors: dict = {}
_tile_list: list = []

def _build_tile_list(vectors):
    def sort_key(stem):
        v = vectors.get(stem, {})
        return (v.get("manually_reviewed", False), v.get("consensus_pct", 0.0))

    stems = sorted(
        (stem for stem in vectors if (tcd_dir / f"{stem}.tif").exists()),
        key=sort_key,
        reverse=(args.sort == "desc"),
    )
    if args.filter == "low":
        stems = [s for s in stems
                 if vectors[s].get("consensus_pct", 100) < args.min_consensus]
    elif args.filter == "high":
        stems = [s for s in stems
                 if (vectors[s].get("consensus_pct", 0) >= args.min_consensus
                     and not vectors[s].get("manually_reviewed", False))]
    elif args.filter == "unreviewed":
        stems = [s for s in stems if not vectors[s].get("manually_reviewed", False)]
    elif args.filter == "reviewed":
        stems = [s for s in stems if vectors[s].get("manually_reviewed", False)]
    return stems

def load_vectors():
    return _vectors

def save_vectors(data):
    json_path.write_text(json.dumps(data, indent=2))

# ── Thumbnail cache ────────────────────────────────────────────────────────────

_thumb_cache: dict = {}

def get_thumbnail(stem):
    if stem in _thumb_cache:
        return _thumb_cache[stem]
    tif_path = tcd_dir / f"{stem}.tif"
    with rasterio.open(tif_path) as src:
        data = src.read()[:3]                        # (3, H, W)
    img = Image.fromarray(np.moveaxis(data, 0, -1))  # (H, W, 3)
    img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    _thumb_cache[stem] = buf.getvalue()
    return _thumb_cache[stem]

# ── Routes ────────────────────────────────────────────────────────────────────

def _finite(val, default=0.0):
    """Return val if it's a finite float, otherwise default."""
    try:
        f = float(val)
        return f if f == f else default   # f != f iff NaN
    except (TypeError, ValueError):
        return default

@app.get("/api/tiles")
def api_tiles():
    out = []
    for s in _tile_list:
        v = _vectors[s]
        out.append({
            "stem":              s,
            "shadow_x":          _finite(v.get("shadow_x")),
            "shadow_y":          _finite(v.get("shadow_y")),
            "shadow_angle_deg":  _finite(v.get("shadow_angle_deg")),
            "consensus_pct":     _finite(v.get("consensus_pct")),
            "circular_std_deg":  _finite(v.get("circular_std_deg")),
            "n_inliers":         int(v.get("n_inliers", 0)),
            "n_crops":           int(v.get("n_crops", 30)),
            "manually_reviewed": bool(v.get("manually_reviewed", False)),
        })
    return jsonify(out)


@app.get("/api/tile_image/<stem>")
def api_tile_image(stem):
    data = get_thumbnail(stem)
    return Response(data, mimetype="image/jpeg")


@app.post("/api/save/<stem>")
def api_save(stem):
    payload = request.json
    if stem not in _vectors:
        return jsonify({"error": "unknown stem"}), 404
    _vectors[stem].update({
        "shadow_x":          float(payload["shadow_x"]),
        "shadow_y":          float(payload["shadow_y"]),
        "shadow_angle_deg":  float(payload["shadow_angle_deg"]),
        "manually_reviewed": bool(payload.get("manually_reviewed", True)),
    })
    save_vectors(_vectors)
    return jsonify({"ok": True})


@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")


# ── UI ────────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TCD Shadow Review</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: monospace; background: #1a1a1a; color: #ddd;
         display: flex; height: 100vh; overflow: hidden; }
  #sidebar { width: 260px; flex-shrink: 0; background: #111; overflow-y: auto;
             border-right: 1px solid #333; }
  #sidebar h2 { padding: 10px; font-size: 13px; color: #888;
                border-bottom: 1px solid #333; }
  .tile-item { padding: 8px 10px; cursor: pointer; border-bottom: 1px solid #222;
               font-size: 11px; }
  .tile-item:hover { background: #222; }
  .tile-item.active { background: #2a3a2a; border-left: 3px solid #4a4; }
  .tile-item .consensus { color: #888; }
  .tile-item.low-conf .consensus { color: #c84; }
  .tile-item.reviewed { opacity: 0.5; }
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #toolbar { padding: 10px 16px; background: #111; border-bottom: 1px solid #333;
             display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  #toolbar h1 { font-size: 14px; color: #aaa; flex: 1; }
  #stats { font-size: 11px; color: #888; }
  #stats span { margin-right: 14px; }
  #stats .hi { color: #6c6; }
  #stats .lo { color: #c84; }
  button { padding: 5px 14px; border: 1px solid #555; background: #2a2a2a;
           color: #ccc; cursor: pointer; font-size: 12px; border-radius: 3px; }
  button:hover { background: #3a3a3a; }
  button.primary { background: #2a4a2a; border-color: #4a4; color: #8f8; }
  button.primary:hover { background: #3a5a3a; }
  #img-wrap { flex: 1; display: flex; align-items: center;
              justify-content: center; overflow: hidden; background: #0d0d0d;
              cursor: crosshair; }
  #img-wrap > div { position: relative; display: inline-block; }
  #tile-img { display: block; max-width: 100%; max-height: calc(100vh - 130px); }
  #arrow-svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
               pointer-events: none; overflow: visible; }
  #hint { padding: 4px 16px; font-size: 11px; color: #555; background: #111;
          border-top: 1px solid #222; }
  #progress { padding: 4px 16px; font-size: 11px; color: #666; background: #111; }
</style>
</head>
<body>

<div id="sidebar">
  <h2 id="filter-label">Loading...</h2>
  <div id="tile-list"></div>
</div>

<div id="main">
  <div id="toolbar">
    <h1 id="tile-name">—</h1>
    <div id="stats">
      <span>Consensus: <b id="s-consensus">—</b></span>
      <span>Inliers: <b id="s-inliers">—</b></span>
      <span>Std: <b id="s-std">—</b>°</span>
      <span>Angle: <b id="s-angle">—</b>°</span>
    </div>
    <button onclick="prevTile()">← Prev</button>
    <button onclick="nextTile()">→ Next</button>
    <button class="primary" onclick="acceptAuto()" title="A">Accept (A)</button>
    <button onclick="saveOverride()" title="Enter">Save override (↵)</button>
    <button onclick="nextTile()" title="S">Skip (S)</button>
  </div>

  <div id="img-wrap">
    <div>
      <img id="tile-img" src="" alt="">
      <svg id="arrow-svg" viewBox="0 0 600 600">
        <circle id="a-dot" cx="300" cy="300" r="5" fill="#f84"/>
        <line   id="a-line" x1="300" y1="300" x2="300" y2="167"
                stroke="#f84" stroke-width="3" stroke-linecap="round"/>
        <polygon id="a-head" points="300,140 288,168 312,168" fill="#f84"/>
        <text    id="a-label" x="8" y="22" font-size="16" fill="white"
                 font-family="monospace" font-weight="bold">0°</text>
      </svg>
    </div>
  </div>

  <div id="progress">— / —</div>
  <pre id="debug" style="padding:4px 14px;font-size:10px;color:#0f0;background:#111;
       max-height:80px;overflow:auto;border-top:1px solid #333;margin:0"></pre>
</div>

<script>
let tiles = [], idx = 0, currentAngle = 0, overridden = false;
const MIN_CONSENSUS = """ + str(args.min_consensus) + """;
const SIDEBAR_WIN   = 80;
const dbg = document.getElementById('debug');
function log(msg) { dbg.textContent += msg + '\\n'; dbg.scrollTop = dbg.scrollHeight; }

const tileImg = document.getElementById('tile-img');
const svg     = document.getElementById('arrow-svg');
const aLine   = document.getElementById('a-line');
const aHead   = document.getElementById('a-head');
const aDot    = document.getElementById('a-dot');
const aLabel  = document.getElementById('a-label');

function drawArrow() {
  const W   = tileImg.naturalWidth  || 600;
  const H   = tileImg.naturalHeight || 600;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const cx  = W / 2, cy = H / 2;
  const len = Math.min(W, H) * 0.22;
  const rad = (currentAngle - 90) * Math.PI / 180;
  const ex  = cx + len * Math.cos(rad);
  const ey  = cy + len * Math.sin(rad);
  const hl  = Math.min(W, H) * 0.04;
  const ha  = 0.4;
  const col = overridden ? '#4af' : '#f84';
  aLine.setAttribute('x1', cx); aLine.setAttribute('y1', cy);
  aLine.setAttribute('x2', ex); aLine.setAttribute('y2', ey);
  aLine.setAttribute('stroke', col);
  aLine.setAttribute('stroke-width', Math.min(W,H)*0.006);
  const hx1 = ex - hl*Math.cos(rad-ha), hy1 = ey - hl*Math.sin(rad-ha);
  const hx2 = ex - hl*Math.cos(rad+ha), hy2 = ey - hl*Math.sin(rad+ha);
  aHead.setAttribute('points', `${ex},${ey} ${hx1},${hy1} ${hx2},${hy2}`);
  aHead.setAttribute('fill', col);
  aDot.setAttribute('cx', cx); aDot.setAttribute('cy', cy);
  aDot.setAttribute('r', Math.min(W,H)*0.008); aDot.setAttribute('fill', col);
  aLabel.setAttribute('x', W*0.015); aLabel.setAttribute('y', H*0.05);
  aLabel.setAttribute('font-size', Math.min(W,H)*0.035);
  aLabel.textContent = currentAngle.toFixed(1) + '°' + (overridden ? ' OVR' : '');
  aLabel.setAttribute('fill', col);
}

document.getElementById('img-wrap').addEventListener('click', (e) => {
  const r  = tileImg.getBoundingClientRect();
  if (!r.width) return;
  const dx = (e.clientX - r.left) - r.width  / 2;
  const dy = (e.clientY - r.top)  - r.height / 2;
  currentAngle = (Math.atan2(dx, -dy) * 180 / Math.PI + 360) % 360;
  document.getElementById('s-angle').textContent = currentAngle.toFixed(1);
  overridden = true;
  drawArrow();
});

async function loadTiles() {
  try {
    log('fetching tiles...');
    const resp = await fetch('/api/tiles');
    log('status: ' + resp.status);
    const text = await resp.text();
    log('body length: ' + text.length);
    tiles = JSON.parse(text);
    log('parsed: ' + tiles.length + ' tiles');
    document.getElementById('filter-label').textContent = tiles.length + ' tiles';
    if (tiles.length) showTile(0);
    else log('no tiles');
  } catch(e) {
    log('ERROR: ' + e.message);
  }
}

function renderSidebar() {
  const ul = document.getElementById('tile-list');
  ul.innerHTML = '';
  const lo = Math.max(0, idx - SIDEBAR_WIN / 2);
  const hi = Math.min(tiles.length, lo + SIDEBAR_WIN);
  if (lo > 0) { const s = document.createElement('div'); s.className='tile-item';
    s.style.color='#555'; s.textContent=`… ${lo} earlier tiles`; ul.appendChild(s); }
  for (let i = lo; i < hi; i++) {
    const t = tiles[i], d = document.createElement('div');
    d.className = 'tile-item'+(t.manually_reviewed?' reviewed':'')+(i===idx?' active':'');
    d.innerHTML = `<div>${t.stem.replace('tcd_tile_','tile ')}</div>` +
      `<div class="consensus">${t.consensus_pct.toFixed(0)}%${t.manually_reviewed?' ✓':''}</div>`;
    d.onclick = () => showTile(i);
    ul.appendChild(d);
  }
  if (hi < tiles.length) { const s = document.createElement('div'); s.className='tile-item';
    s.style.color='#555'; s.textContent=`… ${tiles.length-hi} more tiles`; ul.appendChild(s); }
}

function showTile(i) {
  idx = i; overridden = false;
  const t = tiles[i];
  currentAngle = t.shadow_angle_deg;
  document.getElementById('tile-name').textContent = t.stem;
  const conEl = document.getElementById('s-consensus');
  conEl.textContent = t.consensus_pct.toFixed(1) + '%';
  conEl.className = t.consensus_pct >= MIN_CONSENSUS ? 'hi' : 'lo';
  document.getElementById('s-inliers').textContent = `${t.n_inliers}/${t.n_crops}`;
  document.getElementById('s-std').textContent     = t.circular_std_deg.toFixed(1);
  document.getElementById('s-angle').textContent   = t.shadow_angle_deg.toFixed(1);
  document.getElementById('progress').textContent  = `${i+1} / ${tiles.length}`;
  renderSidebar();
  log(`loading image for ${t.stem}`);
  tileImg.onerror = () => log(`ERROR loading image: ${tileImg.src}`);
  tileImg.onload  = () => { log(`image loaded ${tileImg.naturalWidth}x${tileImg.naturalHeight}`); drawArrow(); };
  tileImg.src     = `/api/tile_image/${t.stem}?v=${i}`;
}

async function saveVector(angle_deg, reviewed) {
  const rad = angle_deg * Math.PI / 180;
  await fetch(`/api/save/${tiles[idx].stem}`, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({shadow_x:Math.sin(rad), shadow_y:Math.cos(rad),
                          shadow_angle_deg:angle_deg, manually_reviewed:reviewed})
  });
  tiles[idx].shadow_angle_deg = angle_deg;
  tiles[idx].manually_reviewed = reviewed;
  renderSidebar();
}

async function acceptAuto()   { await saveVector(tiles[idx].shadow_angle_deg, true); nextTile(); }
async function saveOverride() { await saveVector(currentAngle, true); nextTile(); }
function prevTile() { if (idx > 0)              showTile(idx-1); }
function nextTile() { if (idx < tiles.length-1) showTile(idx+1); }

document.addEventListener('keydown', e => {
  if (e.key==='a'||e.key==='A') acceptAuto();
  if (e.key==='s'||e.key==='S') nextTile();
  if (e.key==='Enter')          saveOverride();
  if (e.key==='ArrowLeft')      prevTile();
  if (e.key==='ArrowRight')     nextTile();
});

loadTiles();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    if not json_path.exists():
        print(f"❌  No shadow vectors JSON found at {json_path}")
        print("    Run estimate_tcd_shadows.py first.")
        sys.exit(1)

    print("Loading vectors...", end=" ", flush=True)
    _vectors.update(json.loads(json_path.read_text()))
    print(f"{len(_vectors)} entries")

    print("Building tile list...", end=" ", flush=True)
    _tile_list.extend(_build_tile_list(_vectors)[:args.max_tiles])
    print(f"{len(_tile_list)} tiles (max={args.max_tiles})")

    print(f"Filter: '{args.filter}', min-consensus={args.min_consensus}%")
    if args.filter == "high":
        print("Tip: sorted by consensus ascending — borderline cases first.")
    print(f"Open http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
