#!/usr/bin/env bash
# Batch Pixelization for b ∈ {2, 4, 8, 16}.
#   ./scripts/run_pixelize.sh                                       # ORL defaults
#   ./scripts/run_pixelize.sh data/facescrub data/deid/fs_pix --detect-faces

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT="${1:-${INPUT:-data/att_faces}}"
OUTPUT_ROOT="${2:-${OUTPUT_ROOT:-data/deid/pixelized}}"
DETECT="${DETECT:-}"
BACKEND="${BACKEND:-haar}"
RUNNER="${RUNNER:-uv run python}"

if [[ ! -d "$INPUT" ]]; then
    echo "[run_pixelize] input not a directory: $INPUT" >&2
    exit 1
fi

echo "[run_pixelize] in=$INPUT out=$OUTPUT_ROOT backend=$BACKEND detect=${DETECT:-no}"
mkdir -p "$OUTPUT_ROOT"
START_TS=$(date +%s)

for B in 2 4 8 16; do
    OUT="$OUTPUT_ROOT/pix_b${B}"
    echo "▶ b=$B → $OUT"
    $RUNNER -m facedeid.pixelize \
        --input "$INPUT" --output "$OUT" \
        --b "$B" --backend "$BACKEND" $DETECT
done

echo "[run_pixelize] done in $(($(date +%s) - START_TS))s"
