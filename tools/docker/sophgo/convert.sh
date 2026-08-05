#!/bin/bash
# 在 tpuc_dev 容器内将 ONNX 转换为 bmodel (支持 BM1688 / CV186AH)
# 用法:
#   docker run --rm -it -v <onnx目录>:/conv tpuc_dev:1.27 bash /conv/convert.sh \
#       --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
#       --chip bm1688 --quantize F16 --out yolo11n_bm1688.bmodel
set -e

ONNX=
NAME=
SHAPES="[[1,3,640,640]]"
CHIP=bm1688
QUANTIZE=F16
OUT=

while [ $# -gt 0 ]; do
    case "$1" in
        --onnx)     ONNX=$2; shift 2 ;;
        --name)     NAME=$2; shift 2 ;;
        --shapes)   SHAPES=$2; shift 2 ;;
        --chip)     CHIP=$2; shift 2 ;;
        --quantize) QUANTIZE=$2; shift 2 ;;
        --out)      OUT=$2; shift 2 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

[ -n "$ONNX" ] || { echo "必须指定 --onnx" >&2; exit 1; }
NAME=${NAME:-$(basename "$ONNX" .onnx)}
OUT=${OUT:-${NAME}_${CHIP}.bmodel}
OUT_DIR=$(dirname "$OUT")

pip3 install --quiet --force-reinstall numpy==1.24.3 2>&1 | tail -1 || true

MT=$(python3 -c "import tpu_mlir,os; print(os.path.join(os.path.dirname(tpu_mlir.__file__),'python','tools','model_transform.py'))")
MD=$(python3 -c "import tpu_mlir,os; print(os.path.join(os.path.dirname(tpu_mlir.__file__),'python','tools','model_deploy.py'))")

mkdir -p "$OUT_DIR"
echo "=== model_transform ($ONNX -> $NAME.mlir) ==="
python3 "$MT" --model_name "$NAME" --model_def "$ONNX" \
    --input_shapes "$SHAPES" --mlir "$OUT_DIR/$NAME.mlir"

echo "=== model_deploy ($QUANTIZE / $CHIP) ==="
python3 "$MD" --mlir "$OUT_DIR/$NAME.mlir" --quantize "$QUANTIZE" --chip "$CHIP" \
    --model "$OUT_DIR/$OUT"

ls -la "$OUT_DIR/$OUT"
echo "转换完成: $OUT_DIR/$OUT"
