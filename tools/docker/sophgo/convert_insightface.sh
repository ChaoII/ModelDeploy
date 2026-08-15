#!/bin/bash
# 批量转换 insightface buffalo_l 5 个模型到 bmodel（BM1688 / CV186AH）。
# 依赖 tools/docker/sophgo/convert.sh（tpuc_dev 容器内运行，F16 即可保证精度）。
#
# 用法（在 tpuc_dev 容器内）:
#   # 把 ONNX 模型目录挂载到 /conv
#   docker run --rm -it -v <onnx目录>:/conv tpuc_dev:1.27 bash /conv/convert_insightface.sh \
#       --onnx_dir /conv --out_dir /conv/sophgo_models --chip bm1688 --quantize F16
#
# 或用顶层转换脚本批量执行:
#   bash convert.sh --onnx det_10g.onnx   --name det_10g   --shapes "[[1,3,640,640]]" --chip bm1688 --quantize F16 --out sophgo/det_10g.bmodel
#   bash convert.sh --onnx 2d106det.onnx  --name 2d106det  --shapes "[[1,3,192,192]]" --chip bm1688 --quantize F16 --out sophgo/2d106det.bmodel
#   bash convert.sh --onnx 1k3d68.onnx    --name 1k3d68    --shapes "[[1,3,192,192]]" --chip bm1688 --quantize F16 --out sophgo/1k3d68.bmodel
#   bash convert.sh --onnx w600k_r50.onnx --name w600k_r50 --shapes "[[1,3,112,112]]" --chip bm1688 --quantize F16 --out sophgo/w600k_r50.bmodel
#   bash convert.sh --onnx genderage.onnx --name genderage --shapes "[[1,3,96,96]]"   --chip bm1688 --quantize F16 --out sophgo/genderage.bmodel
#
# 注: det_10g 动态 batch（shape [0,3,640,640]）在 BM1688 上需固定 batch=1。
#     若需 INT8，加 --cali_images <图片目录>（人脸图，数量建议 >=100）。
set -e

ONNX_DIR=
OUT_DIR=
CHIP=bm1688
QUANTIZE=F16
CONVERT=/conv/convert.sh

while [ $# -gt 0 ]; do
    case "$1" in
        --onnx_dir) ONNX_DIR=$2; shift 2 ;;
        --out_dir)  OUT_DIR=$2; shift 2 ;;
        --chip)     CHIP=$2; shift 2 ;;
        --quantize) QUANTIZE=$2; shift 2 ;;
        --convert)  CONVERT=$2; shift 2 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

[ -n "$ONNX_DIR" ] || { echo "必须指定 --onnx_dir" >&2; exit 1; }
OUT_DIR=${OUT_DIR:-"$ONNX_DIR/sophgo_models"}
mkdir -p "$OUT_DIR"

declare -A SHAPES=(
    [det_10g]=640
    [2d106det]=192
    [1k3d68]=192
    [w600k_r50]=112
    [genderage]=96
)

for name in det_10g 2d106det 1k3d68 w600k_r50 genderage; do
    size=${SHAPES[$name]}
    onnx="$ONNX_DIR/$name.onnx"
    if [ ! -f "$onnx" ]; then
        echo "跳过: 未找到 $onnx" >&2
        continue
    fi
    echo "=== 转换 $name (1x3x${size}x${size}) ==="
    bash "$CONVERT" --onnx "$onnx" --name "$name" \
        --shapes "[[1,3,${size},${size}]]" \
        --chip "$CHIP" --quantize "$QUANTIZE" --out "$OUT_DIR/$name.bmodel"
done

echo "全部完成: $OUT_DIR"
ls -la "$OUT_DIR"
