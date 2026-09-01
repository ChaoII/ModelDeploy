#!/bin/bash
# 在 tpuc_dev 容器内将 ONNX 转换为 bmodel (支持 BM1688 / CV186AH)
# 支持 F16 / BF16 / INT8（INT8 需校准，见 --cali_* 参数）
# 用法:
#   docker run --rm -it -v <onnx目录>:/conv tpuc_dev:1.27 bash /conv/convert.sh \
#       --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
#       --chip bm1688 --quantize F16 --out yolo11n_bm1688.bmodel
#
#   # INT8（自动从图片目录生成校准数据）
#   docker run --rm -it -v <onnx目录>:/conv -v <图片目录>:/cali_img tpuc_dev:1.27 bash /conv/convert.sh \
#       --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
#       --chip bm1688 --quantize INT8 --cali_images /cali_img --cali_num 100 \
#       --out yolo11n_bm1688_int8.bmodel
#
#   # INT8（直接给已归一化的 npy 列表）
#   docker run --rm -it -v <onnx目录>:/conv -v <npy目录>:/cali_data tpuc_dev:1.27 bash /conv/convert.sh \
#       --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
#       --chip bm1688 --quantize INT8 --cali_data_list /cali_data/data_list.txt \
#       --out yolo11n_bm1688_int8.bmodel
set -e

ONNX=
NAME=
SHAPES="[[1,3,640,640]]"
CHIP=bm1688
QUANTIZE=F16
OUT=
CALI_IMAGES=
CALI_DATA_LIST=
CALI_NUM=100
CALI_METHOD=kl
QTABLE=

while [ $# -gt 0 ]; do
    case "$1" in
        --onnx)          ONNX=$2; shift 2 ;;
        --name)          NAME=$2; shift 2 ;;
        --shapes)        SHAPES=$2; shift 2 ;;
        --chip)          CHIP=$2; shift 2 ;;
        --quantize)      QUANTIZE=$2; shift 2 ;;
        --out)           OUT=$2; shift 2 ;;
        --cali_images)   CALI_IMAGES=$2; shift 2 ;;
        --cali_data_list) CALI_DATA_LIST=$2; shift 2 ;;
        --cali_num)      CALI_NUM=$2; shift 2 ;;
        --cali_method)   CALI_METHOD=$2; shift 2 ;;
        --qtable)        QTABLE=$2; shift 2 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

[ -n "$ONNX" ] || { echo "必须指定 --onnx" >&2; exit 1; }
if [[ "$ONNX" != /* ]]; then
    ONNX="/conv/$ONNX"   # 相对路径默认指向挂载目录 /conv
fi
NAME=${NAME:-$(basename "$ONNX" .onnx)}
if [ -z "$OUT" ]; then
    OUT="${NAME}_${CHIP}.bmodel"
elif [[ "$OUT" != /* ]]; then
    OUT="/conv/$OUT"   # 相对路径默认落在挂载目录 /conv
fi
OUT_DIR=$(dirname "$OUT")
mkdir -p "$OUT_DIR"
# 关键：统一在输出目录下运行，保证 model_transform 生成的
# *_top_f32_all_weight.npz（写到 cwd）与 mlir/校准表/产物同目录，部署时相对名可解析
cd "$OUT_DIR"

# tpu-mlir 1.27 不兼容 numpy 2.x；仅当 numpy 为 2.x 时才降级（1.x 直接跳过，避免每次联网重装挂死）
if python3 -c "import numpy as np; import sys; sys.exit(0 if np.__version__.startswith('1.') else 1)" 2>/dev/null; then
    echo "numpy 1.x 已满足，跳过重装"
else
    pip3 install --quiet --force-reinstall --no-cache-dir numpy==1.24.3 2>&1 | tail -1 || true
fi

MT=$(python3 -c "import tpu_mlir,os; print(os.path.join(os.path.dirname(tpu_mlir.__file__),'python','tools','model_transform.py'))")
RC=$(python3 -c "import tpu_mlir,os; print(os.path.join(os.path.dirname(tpu_mlir.__file__),'python','tools','run_calibration.py'))")
MD=$(python3 -c "import tpu_mlir,os; print(os.path.join(os.path.dirname(tpu_mlir.__file__),'python','tools','model_deploy.py'))")
# pymlir (*.so) 在 tpu_mlir/python 下,需加入 PYTHONPATH 才能 import
_TPU_PY=$(python3 -c "import tpu_mlir,os; print(os.path.join(os.path.dirname(tpu_mlir.__file__),'python'))")
export PYTHONPATH="$_TPU_PY:${PYTHONPATH:-}"

echo "=== 1/3 model_transform ($ONNX -> $NAME.mlir) ==="
python3 "$MT" --model_name "$NAME" --model_def "$ONNX" \
    --input_shapes "$SHAPES" --mlir "$NAME.mlir"
ls -la "${NAME}_top_f32_all_weight.npz" 2>&1

CALI_TABLE="${NAME}_cali_table.txt"
if [ "$QUANTIZE" = "INT8" ]; then
    echo "=== 2/3 run_calibration (INT8, method=$CALI_METHOD) ==="
    # 生成校准数据列表
    if [ -n "$CALI_DATA_LIST" ]; then
        DATA_LIST="$CALI_DATA_LIST"
    elif [ -n "$CALI_IMAGES" ]; then
        WORK_DIR="_cali_${NAME}"
        mkdir -p "$WORK_DIR"
        # 从图片目录选 CALI_NUM 张，resize 到 bmodel 输入尺寸并归一化到 [0,1]
        python3 - "$CALI_IMAGES" "$WORK_DIR" "$SHAPES" "$CALI_NUM" "${NAME}_data_list.txt" <<'EOF'
import os, sys, glob, numpy as np
from PIL import Image

cali_dir, work_dir, shapes, cali_num, list_out = sys.argv[1:]
shape = eval(shapes)[0]  # [1,3,H,W]
_, C, H, W = shape
imgs = sorted(glob.glob(os.path.join(cali_dir, "*.*")))
imgs = [p for p in imgs if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
imgs = imgs[:int(cali_num)]
assert imgs, f"未在 {cali_dir} 找到图片"

data_list = []
for i, p in enumerate(imgs):
    im = Image.open(p).convert("RGB").resize((W, H))
    arr = np.array(im).astype(np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)[np.newaxis]  # [1,C,H,W]
    out = os.path.join(work_dir, f"cali_{i}.npy")
    np.save(out, chw)
    data_list.append(out)
with open(list_out, "w") as f:
    f.write("\n".join(data_list))
print(f"生成 {len(data_list)} 份校准数据 -> {list_out}")
EOF
        DATA_LIST="${NAME}_data_list.txt"
    else
        echo "INT8 需提供 --cali_images <图片目录> 或 --cali_data_list <npy列表>" >&2
        exit 1
    fi

    python3 "$RC" "$NAME.mlir" --data_list "$DATA_LIST" \
        --cali_method "$CALI_METHOD" -o "$CALI_TABLE"
    ls -la "$CALI_TABLE"
fi

echo "=== 3/3 model_deploy ($QUANTIZE / $CHIP) ==="
if [ "$QUANTIZE" = "INT8" ]; then
    if [ -n "$QTABLE" ]; then
        echo "  -> 混合精度量化表: $QTABLE"
        python3 "$MD" --mlir "$NAME.mlir" --quantize INT8 \
            --calibration_table "$CALI_TABLE" --quantize_table "$QTABLE" \
            --chip "$CHIP" --model "$OUT"
    else
        python3 "$MD" --mlir "$NAME.mlir" --quantize INT8 \
            --calibration_table "$CALI_TABLE" --chip "$CHIP" --model "$OUT"
    fi
else
    python3 "$MD" --mlir "$NAME.mlir" --quantize "$QUANTIZE" \
        --chip "$CHIP" --model "$OUT"
fi

ls -la "$OUT"
echo "转换完成: $OUT"
