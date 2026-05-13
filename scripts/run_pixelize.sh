#!/usr/bin/env bash
# run_pixelize.sh
# ---------------
# 批次對指定資料集套用 Pixelization,b ∈ {2, 4, 8, 16},產出 4 組去識別化資料集。
#
# 使用方式(在專案根目錄執行):
#   ./scripts/run_pixelize.sh                     # input=data/att_faces, output_root=data/deid/pixelized
#   ./scripts/run_pixelize.sh data/facescrub data/deid/fs_pix --detect-faces
#   INPUT=data/celeb OUTPUT_ROOT=data/deid/celeb_pix DETECT=--detect-faces ./scripts/run_pixelize.sh
#
# 對 ORL 不要加 --detect-faces(整張就是臉,直接全圖 pixelize 即可)。
# 對 FaceScrub / CelebA 等含背景的資料集才需要 --detect-faces。
#
# 產出:$OUTPUT_ROOT/pix_b2/  pix_b4/  pix_b8/  pix_b16/   (對應 input 結構,所有圖片改成 .png)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT="${1:-${INPUT:-data/att_faces}}"
OUTPUT_ROOT="${2:-${OUTPUT_ROOT:-data/deid/pixelized}}"
DETECT="${DETECT:-}"               # 設成 "--detect-faces" 來啟用人臉偵測
BACKEND="${BACKEND:-haar}"         # haar | hog
RUNNER="${RUNNER:-uv run python}"  # 沒裝 uv 時可改成 RUNNER="python3"(需先 PYTHONPATH=src)

if [[ ! -d "$INPUT" ]]; then
    echo "[run_pixelize] 錯誤:輸入資料夾不存在:$INPUT" >&2
    exit 1
fi

echo "============================================================"
echo "  Pixelization 批次處理"
echo "  input        : $INPUT"
echo "  output_root  : $OUTPUT_ROOT"
echo "  detect-faces : ${DETECT:-(no)}   backend: $BACKEND"
echo "============================================================"

mkdir -p "$OUTPUT_ROOT"
START_TS=$(date +%s)

for B in 2 4 8 16; do
    OUT="$OUTPUT_ROOT/pix_b${B}"
    echo
    echo "▶ b = $B  →  $OUT"
    $RUNNER -m facedeid.pixelize \
        --input "$INPUT" \
        --output "$OUT" \
        --b "$B" \
        --backend "$BACKEND" \
        $DETECT
done

END_TS=$(date +%s)
echo
echo "============================================================"
echo "  全部完成,耗時 $((END_TS - START_TS)) 秒。輸出在:$OUTPUT_ROOT"
echo "============================================================"
