#!/bin/bash
# Wiring smoke: tiny train (test-split stand-in) + tiny eval. Validates ckpt load,
# model build, data pipeline, train step, and eval — no multi-GB train download.
set -e
cd "$(dirname "$0")"
PY=${PY:-/tmp/segf_venv/bin/python}
echo "=== SMOKE: train ==="
"$PY" train.py --name semseg_smoke --smoke --fresh
echo "=== SMOKE: eval ==="
"$PY" eval.py --name semseg_smoke --weights best.pt --limit 4
echo "=== SMOKE OK ==="
