# Sophgo bmodel 转换工具（tpu-mlir）

将 ONNX 模型转换为算能（Sophgo）TPU 可加载的 `.bmodel`（已验证 BM1688，CV186AH 同流程）。

## 文件

| 文件 | 说明 |
|------|------|
| `Dockerfile` | tpu-mlir 1.27 转换环境镜像（python:3.10） |
| `build_docker.sh` | 构建镜像 `tpuc_dev:1.27`（需先放入 `tpu_mlir-1.27-py3-none-any.whl` 与 `tpu-mlir-resource.tar`，从算能官方 Sophon SDK 获取） |
| `convert.sh` | 容器内 ONNX → bmodel 转换脚本 |

## 用法

```bash
# 1. 准备资源并构建镜像
#    将 tpu_mlir-1.27-py3-none-any.whl、tpu-mlir-resource.tar 放入本目录
./build_docker.sh

# 2. 转换（挂载 onnx 所在目录到 /conv）
docker run --rm -it -v <onnx目录>:/conv tpuc_dev:1.27 bash /conv/convert.sh \
    --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
    --chip bm1688 --quantize F16 --out yolo11n_bm1688.bmodel
```

`--quantize` 支持 `F16`（默认推荐）/`BF16`/`INT8`（需 cali_table）；`--chip` 支持 `bm1688`/`cv186x`。

## 注意

1. **tpu-mlir 1.27 对带 NMS 的 ONNX 有 Gather 算子转换 bug**：转换前先用 onnxsim 或脚本把 NMS 从图中裁剪掉，输出原始检测头（如 84×8400 或 5×33600），NMS 由 ModelDeploy SDK 侧 `run_without_nms`（含 sigmoid + 无效框过滤）完成。
2. 无 NMS 模型用 **SDK 默认预处理（letterbox + `/255` 归一化到 `[0,1]`）**，无需调用 `set_normalize(false)`；置信度阈值建议 0.5 以上（0.25 会带出大量低分候选）。
3. bmodel 输入尺寸由 `--shapes` 固定，SDK 端需 `preprocessor.set_size(...)` 与之匹配。
4. 转换容器首次使用若 numpy 报 `core.multiarray failed to import`，先 `pip3 install --force-reinstall --no-cache-dir numpy==1.24.3`（tpu-mlir 1.27 不兼容 numpy 2.x）。
