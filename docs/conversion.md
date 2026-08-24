# ModelDeploy 模型转换与量化

汇总将模型转换/量化为各后端可用格式的完整方法。四种后端与模型格式详见 [后端详解](./backends.md)；算能 TPU 交叉编译与部署实战见 [Sophgo 说明](./sophgo_cross_build_and_test.md)。

## 1. OnnxRuntime 混合精度

将 fp32 模型转为 fp16（内部节点 fp16，输入输出保持 fp32）：

```python
import onnx
from onnxconverter_common import float16

model = onnx.load("model_fp32.onnx")
model_mixed = float16.convert_float_to_float16(model, keep_io_types=True)
onnx.save(model_mixed, "model_mixed.onnx")
```

> 提示：GPU 推理用 OnnxRuntime 时，建议直接用 TRT provider（`enable_trt`/trtexec 生成 fp16 engine），通常更省事、效果更好。

## 2. 动态量化减小体积

uint8 动态量化（仅减小体积，非精度提升）：

```python
from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic(
    model_input="model_fp32.onnx",
    reduce_range=True,
    model_output="model_quant_dynamic.onnx",
    per_channel=True,
    weight_type=QuantType.QUInt8,
)
```

## 3. TRT engine 生成

用 `trtexec` 把 ONNX 编译成 `.engine`（在线构建较慢，建议预生成）：

```bash
trtexec --onnx=yolo11n.onnx ^
        --saveEngine=yolo11n_dyn.engine ^
        --fp16 ^
        --minShapes=images:1x3x320x320 ^
        --optShapes=images:1x3x640x640 ^
        --maxShapes=images:4x3x1280x1280
```

动态输入模型必须给出 `--minShapes/--optShapes/--maxShapes`。加载与配置见 [后端详解-TRT](./backends.md#3-tensorrt-后端)。

## 4. bmodel 生成（算能 Sophgo TPU）

基于 tpu-mlir 将 ONNX 转 `.bmodel`，工具见 [`tools/docker/sophgo/`](../tools/docker/sophgo)。先准备 tpu-mlir 1.27 环境：

```bash
cd tools/docker/sophgo
./build_docker.sh   # 构建 tpuc_dev:1.27
```

F16（精度无损，简单）：

```bash
docker run --rm -it -v <onnx目录>:/conv tpuc_dev:1.27 bash /conv/convert.sh \
    --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
    --chip bm1688 --quantize F16 --out yolo11n_bm1688.bmodel
```

INT8（体积 ~25% 更小、TPU 上快 3~5 倍，需校准）：

```bash
docker run --rm -it \
    -v <onnx目录>:/conv -v <校准图片目录>:/cali_img \
    tpuc_dev:1.27 bash /conv/convert.sh \
    --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
    --chip bm1688 --quantize INT8 --cali_images /cali_img --cali_num 100 \
    --out yolo11n_bm1688_int8.bmodel
```

INT8 + 混合精度表（检测头 score 通道被压死时用，保 score 尾层 F16）：

```bash
docker run --rm -it \
    -v <onnx目录>:/conv -v <校准图片目录>:/cali_img -v tools/docker/sophgo:/tpuconf \
    tpuc_dev:1.27 bash /conv/convert.sh \
    --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
    --chip bm1688 --quantize INT8 --cali_images /cali_img --cali_num 100 \
    --qtable /tpuconf/qtable_f16.txt \
    --out yolo11n_bm1688_int8.bmodel
```

**关键注意：**

1. tpu-mlir 1.27 对带 NMS 的 ONNX 有转换 bug，**转换前务必去掉 NMS**（NMS 由 SDK 侧 `run_without_nms` 完成）。
2. 无 NMS 模型保持 SDK 默认预处理（letterbox + `/255` 到 `[0,1]`），**不要** `set_normalize(false)`；无 NMS 模型置信度阈值建议取 0.5 以上。
3. bmodel 输入尺寸由 `--shapes` 固定，SDK 端 `preprocessor.set_size(...)` 必须匹配。
4. INT8 校准需 50~200 张有代表性图片（`--cali_method` 可选 kl/mse/max）。
5. 各任务（det/cls/obb/seg/pose/sem/depth）的完整转换命令与 INT8 精度结论见 [后端详解-Sophgo](./backends.md#5-sophgo-后端算能-tpu)。
