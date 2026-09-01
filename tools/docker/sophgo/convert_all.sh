#!/bin/bash
# 全模型 ONNX -> bmodel 批量转换（BM1688 / CV186AH），覆盖所有 vision 模型。
# 依赖 tools/docker/sophgo/convert.sh（tpuc_dev 容器内运行）。
#
# 注意：本脚本为遗留批处理（平铺命名/旧产物）。规范转换入口见
# tools/convert/models.json（单源注册表）+ tools/convert/convert_bmodel.ps1
# （镜像目录 + <name>-int8.bmodel 命名，int8 + qtable 混合精度）。
#
# 用法（在 tpuc_dev 容器内）:
#   docker run --rm -it -v <onnx目录>:/conv -v <图片目录>:/cali_img tpuc_dev:1.27 \
#       bash /conv/convert_all.sh --onnx_dir /conv --out_dir /conv/sophgo_models \
#       --chip bm1688 --quantize F16
#   # INT8（需校准图）:
#       ... --quantize INT8 --cali_images /cali_img --cali_num 100
#
# 说明:
#   - F16 默认，保证精度；INT8 提供 ~2x 吞吐，需校准数据。
#   - 每个模型固定输入 shape（batch=1）。det 类动态输入在 BM1688 上固定。
#   - 转换产物命名: <name>_<quant>.bmodel，与 tests/ 的 sophgo 用例约定一致。
set -e

ONNX_DIR=
OUT_DIR=
CHIP=bm1688
QUANTIZE=F16
CONVERT=/conv/convert.sh
CALI_IMAGES=
CALI_NUM=100

while [ $# -gt 0 ]; do
    case "$1" in
        --onnx_dir)     ONNX_DIR=$2; shift 2 ;;
        --out_dir)      OUT_DIR=$2; shift 2 ;;
        --chip)         CHIP=$2; shift 2 ;;
        --quantize)     QUANTIZE=$2; shift 2 ;;
        --convert)      CONVERT=$2; shift 2 ;;
        --cali_images)  CALI_IMAGES=$2; shift 2 ;;
        --cali_num)     CALI_NUM=$2; shift 2 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

[ -n "$ONNX_DIR" ] || { echo "必须指定 --onnx_dir" >&2; exit 1; }
OUT_DIR=${OUT_DIR:-"$ONNX_DIR/sophgo_models"}
mkdir -p "$OUT_DIR"

# name -> "相对路径:输入名:1x3xHxW"
# 固定 batch=1；rec/cls 类动态宽固定到常见尺寸。
declare -A SPECS=(
    # ---- yolo11n 系列（内嵌 NMS 导出无法直接转，用非 NMS 版 + C++ 后处理）----
    [yolo11n_det]="yolo11n/yolo11n_nms.onnx:images:1x3x640x640"
    [yolo11n_cls]="yolo11n/yolo11n-cls.onnx:images:1x3x224x224"
    [yolo11n_obb]="yolo11n/yolo11n-obb.onnx:images:1x3x640x640"
    [yolo11n_pose]="yolo11n/yolo11n-pose.onnx:images:1x3x640x640"
    [yolo11n_seg]="yolo11n/yolo11n-seg.onnx:images:1x3x640x640"
    # ---- face ----
    [scrfd]="seetaface/scrfd_2.5g_bnkps_shape640x640.onnx:input.1:1x3x640x640"
    [age_predictor]="seetaface/age_predictor.onnx:input:1x3x112x112"
    [gender_predictor]="seetaface/gender_predictor.onnx:input:1x3x112x112"
    [face_recognizer]="seetaface/face_recognizer.onnx:input:1x3x248x248"
    [fas_first]="seetaface/fas_first.onnx:_input_8:1x3x224x224"
    [fas_second]="seetaface/fas_second.onnx:_input_151:1x3x224x224"
    # ---- lpr ----
    [yolov5plate]="yolov5plate.onnx:input:1x3x640x640"
    [plate_recognition]="plate_recognition_color.onnx:images:1x3x48x168"
    # ---- ocr（动态输入固定到常用尺寸）----
    [ocr_det]="ocr/ppocrv4_mobile/det_infer.onnx:x:1x3x960x960"
    [ocr_rec]="ocr/ppocrv4_mobile/rec_infer.onnx:x:1x3x48x320"
    [ocr_cls]="ocr/ppocrv4_mobile/cls_infer.onnx:x:1x3x48x192"
    # ---- insightface ----
    [det_10g]="insightface/buffalo_l/det_10g.onnx:data:1x3x640x640"
    [2d106det]="insightface/buffalo_l/2d106det.onnx:data:1x3x192x192"
    [1k3d68]="insightface/buffalo_l/1k3d68.onnx:data:1x3x192x192"
    [w600k_r50]="insightface/buffalo_l/w600k_r50.onnx:input.1:1x3x112x112"
    [genderage]="insightface/buffalo_l/genderage.onnx:data:1x3x96x96"
)

for name in "${!SPECS[@]}"; do
    IFS=: read -r rel onnx_input shape <<< "${SPECS[$name]}"
    onnx="$ONNX_DIR/$rel"
    if [ ! -f "$onnx" ]; then
        echo "跳过: 未找到 $onnx" >&2
        continue
    fi
    out="$OUT_DIR/${name}_${QUANTIZE,,}.bmodel"
    echo "=== 转换 $name ($shape) ==="
    ARGS=(--onnx "$onnx" --name "$name" --shapes "[[$shape]]" \
          --chip "$CHIP" --quantize "$QUANTIZE" --out "$out")
    if [ "$QUANTIZE" = "INT8" ]; then
        if [ -n "$CALI_IMAGES" ]; then
            ARGS+=(--cali_images "$CALI_IMAGES" --cali_num "$CALI_NUM")
        else
            echo "INT8 需 --cali_images <图片目录>" >&2
            exit 1
        fi
    fi
    bash "$CONVERT" "${ARGS[@]}"
done

echo "全部完成: $OUT_DIR"
ls -la "$OUT_DIR"
