#!/usr/bin/env bash
# run_blur.sh
# -----------
# 批次對指定資料集套用 Gaussian Blurring,k ∈ {15, 45, 99},產出 3 組去識別化資料集。
#
# 使用方式(在專案根目錄執行):
#   ./scripts/run_blur.sh                         # input=data/att_faces, output_root=outputs/blurred
#   ./scripts/run_blur.sh data/facescrub outputs/fs_blur --detect-faces
#   INPUT=data/celeb OUTPUT_ROOT=outputs/celeb_blur DETECT=--detect-faces ./scripts/run_blur.sh
#
# 對 ORL 不要加 --detect-faces(整張就是臉,直接全圖 blur,等同論文設定)。
# 對 FaceScrub / CelebA 等含背景的資料集才需要 --detect-faces。
#
# 產出:$OUTPUT_ROOT/blur_k15/  blur_k45/  blur_k99/   (對應 input 結構,所有圖片改成 .png)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT="${1:-${INPUT:-data/att_faces}}"
OUTPUT_ROOT="${2:-${OUTPUT_ROOT:-outputs/blurred}}"
DETECT="${DETECT:-}"               # 設成 "--detect-faces" 來啟用人臉偵測
BACKEND="${BACKEND:-haar}"         # haar | hog
SIGMA="${SIGMA:-0}"                # 0 = 讓 OpenCV 依 k 自動推算 σ
RUNNER="${RUNNER:-uv run python}"  # 沒裝 uv 時可改成 RUNNER="python3"(需先 PYTHONPATH=src)

if [[ ! -d "$INPUT" ]]; then
    echo "[run_blur] 錯誤:輸入資料夾不存在:$INPUT" >&2
    exit 1
fi

echo "============================================================"
echo "  Gaussian Blurring 批次處理"
echo "  input        : $INPUT"
echo "  output_root  : $OUTPUT_ROOT"
echo "  detect-faces : ${DETECT:-(no)}"
echo "  backend      : $BACKEND   sigma: ${SIGMA} (0 = auto)"
echo "============================================================"

mkdir -p "$OUTPUT_ROOT"
START_TS=$(date +%s)

for K in 15 45 99; do
    OUT="$OUTPUT_ROOT/blur_k${K}"
    echo
    echo "▶ k = $K  →  $OUT"
    $RUNNER -m facedeid.gaussian_blur \
        --input "$INPUT" \
        --output "$OUT" \
        --k "$K" \
        --sigma "$SIGMA" \
        --backend "$BACKEND" \
        $DETECT
done

END_TS=$(date +%s)
echo
echo "============================================================"
echo "  全部完成,耗時 $((END_TS - START_TS)) 秒。輸出在:$OUTPUT_ROOT"
echo "============================================================"
