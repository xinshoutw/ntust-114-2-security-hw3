#!/usr/bin/env bash
# Batch Gaussian Blurring for k ∈ {15, 45, 99}.
#   ./scripts/run_blur.sh                                          # ORL defaults
#   ./scripts/run_blur.sh data/facescrub data/deid/fs_blur --detect-faces

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT="${1:-${INPUT:-data/att_faces}}"
OUTPUT_ROOT="${2:-${OUTPUT_ROOT:-data/deid/blurred}}"
DETECT="${DETECT:-}"
BACKEND="${BACKEND:-haar}"
SIGMA="${SIGMA:-0}"
RUNNER="${RUNNER:-uv run python}"

if [[ ! -d "$INPUT" ]]; then
    echo "[run_blur] input not a directory: $INPUT" >&2
    exit 1
fi

echo "[run_blur] in=$INPUT out=$OUTPUT_ROOT backend=$BACKEND sigma=$SIGMA detect=${DETECT:-no}"
mkdir -p "$OUTPUT_ROOT"
START_TS=$(date +%s)

for K in 15 45 99; do
    OUT="$OUTPUT_ROOT/blur_k${K}"
    echo "▶ k=$K → $OUT"
    $RUNNER -m facedeid.gaussian_blur \
        --input "$INPUT" --output "$OUT" \
        --k "$K" --sigma "$SIGMA" --backend "$BACKEND" $DETECT
done

echo "[run_blur] done in $(($(date +%s) - START_TS))s"
