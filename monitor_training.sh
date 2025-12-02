#!/bin/bash
# monitor_training.sh - Track validation AP during training
# Usage: ./monitor_training.sh <path_to_log_file>

if [ -z "$1" ]; then
    echo "Usage: $0 <log_file>"
    echo "Example: $0 train_outputs/run_delta/log.txt"
    exit 1
fi

LOG_FILE=$1

echo "=== Validation AP Progression ==="
echo "Iter  | segm AP | segm AP50 | bbox AP | bbox AP50"
echo "------|---------|-----------|---------|----------"

# Extract validation metrics from log
grep -A 20 "Evaluation results for segm:" "$LOG_FILE" | \
    awk '
    /iter:/ {iter=$4}
    /copypaste: [0-9]/ && /segm/ {
        split($0, a, ",")
        segm_ap = a[1]
        segm_ap50 = a[2]
        gsub(/.*copypaste: /, "", segm_ap)
    }
    /copypaste: [0-9]/ && /bbox/ {
        split($0, a, ",")
        bbox_ap = a[1]
        bbox_ap50 = a[2]
        gsub(/.*copypaste: /, "", bbox_ap)
        printf "%s | %s | %s | %s | %s\n", iter, segm_ap, segm_ap50, bbox_ap, bbox_ap50
    }
    ' | tail -15

echo ""
echo "=== Latest Metrics ==="
tail -1 "$LOG_FILE" | grep -o "iter: [0-9]*" || echo "Training in progress..."
grep "segm AP50" "$LOG_FILE" | tail -1 || echo "No validation yet"
