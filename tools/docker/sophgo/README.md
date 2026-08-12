# Sophgo bmodel 转换工具（tpu-mlir）

将 ONNX 模型转换为算能（Sophgo）TPU 可加载的 `.bmodel`（已验证 BM1688，CV186AH 同流程）。

## 文件

| 文件 | 说明 |
|------|------|
| `Dockerfile` | tpu-mlir 1.27 转换环境镜像（python:3.10） |
| `build_docker.sh` | 构建镜像 `tpuc_dev:1.27`（需先放入 `tpu_mlir-1.27-py3-none-any.whl` 与 `tpu-mlir-resource.tar`，从算能官方 Sophon SDK 获取） |
| `convert.sh` | 容器内 ONNX → bmodel 转换脚本（F16/BF16/INT8） |

## 用法

### 准备环境

```bash
# 将 tpu_mlir-1.27-py3-none-any.whl、tpu-mlir-resource.tar 放入本目录
./build_docker.sh
```

### F16 / BF16（无需校准）

```bash
docker run --rm -it -v <onnx目录>:/conv tpuc_dev:1.27 bash /conv/convert.sh \
    --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
    --chip bm1688 --quantize F16 --out yolo11n_bm1688.bmodel
```

### INT8（需校准，推荐）

```bash
# 方式 A：从图片目录自动生成校准数据（--cali_num 张，resize 到输入尺寸并 /255 归一化）
docker run --rm -it \
    -v <onnx目录>:/conv -v <校准图片目录>:/cali_img \
    tpuc_dev:1.27 bash /conv/convert.sh \
    --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
    --chip bm1688 --quantize INT8 --cali_images /cali_img --cali_num 100 \
    --out yolo11n_bm1688_int8.bmodel

# 方式 B：直接提供已预处理（[1,3,H,W] float32, 值域 [0,1]）的 npy 列表
docker run --rm -it \
    -v <onnx目录>:/conv -v <npy目录>:/cali_data \
    tpuc_dev:1.27 bash /conv/convert.sh \
    --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
    --chip bm1688 --quantize INT8 --cali_data_list /cali_data/data_list.txt \
    --out yolo11n_bm1688_int8.bmodel

# 方式 C：INT8 + 混合精度量化表（推荐，解决检测头置信度被量化压死的问题）
#   当模型输出形如 [B,5,N]=[cx,cy,w,h,score]，纯 INT8 会把 score 通道(0~1)压成全 0
#   导致无检出。用 --qtable 让置信度相关尾部算子保持 F16（其余仍 INT8），
#   见 qtable_f16.txt 注释。先转一次后用 `grep 'loc(' *.mlir` 确认算子名。
docker run --rm -it \
    -v <onnx目录>:/conv -v <校准图片目录>:/cali_img \
    -v <本目录>:/tpuconf \
    tpuc_dev:1.27 bash /conv/convert.sh \
    --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" \
    --chip bm1688 --quantize INT8 --cali_images /cali_img --cali_num 100 \
    --qtable /tpuconf/qtable_f16.txt \
    --out yolo11n_bm1688_int8.bmodel
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--onnx` | 必填 | ONNX 模型路径（相对路径默认在挂载目录 `/conv`） |
| `--name` | onnx 文件名 | 模型名（mlir/校准表前缀） |
| `--shapes` | `[[1,3,640,640]]` | 输入形状（bmodel 固定输入） |
| `--chip` | `bm1688` | `bm1688` / `cv186x` |
| `--quantize` | `F16` | `F16` / `BF16` / `INT8`（INT8 需校准参数） |
| `--out` | `${name}_${chip}.bmodel` | 输出 bmodel 路径（相对路径默认在 `/conv`） |
| `--cali_images` | — | 校准图片目录（INT8 用，自动生成 npy） |
| `--cali_data_list` | — | 已预处理 npy 列表文件（INT8 用，每行一个 npy） |
| `--cali_num` | `100` | 校准图片数量 |
| `--cali_method` | `kl` | 校准方法（kl/mse/max/percentile9999/aciq_gauss 等） |
| `--qtable` | — | 混合精度量化表（INT8 可选，见 `qtable_f16.txt`） |

`--quantize INT8` 时 `--cali_images` 与 `--cali_data_list` 二选一（均缺省则报错）。

## 注意

1. **tpu-mlir 1.27 对带 NMS 的 ONNX 有 Gather 算子转换 bug**：转换前先用 onnxsim 或脚本把 NMS 从图中裁剪掉，输出原始检测头（如 84×8400 或 5×33600），NMS 由 ModelDeploy SDK 侧 `run_without_nms`（含 sigmoid + 无效框过滤）完成。
2. 无 NMS 模型用 **SDK 默认预处理（letterbox + `/255` 归一化到 `[0,1]`）**，无需调用 `set_normalize(false)`；置信度阈值建议 0.5 以上（0.25 会带出大量低分候选）。
3. bmodel 输入尺寸由 `--shapes` 固定，SDK 端需 `preprocessor.set_size(...)` 与之匹配。
4. 转换容器首次使用若 numpy 报 `core.multiarray failed to import`，先 `pip3 install --force-reinstall --no-cache-dir numpy==1.24.3`（tpu-mlir 1.27 不兼容 numpy 2.x，convert.sh 已自动处理）。
5. **INT8 校准的 mlir 必须与其配套权重文件一起生成**：`model_transform` 会把 `${name}_top_f32_all_weight.npz` 写到**当前工作目录**，因此 convert.sh 统一 `cd` 到输出目录执行三步（transform/校准/部署），勿手工混用不同 mlir 与 npz。
6. **检测头置信度通道被量化压死**：若输出形如 `[B,5,N]=[cx,cy,w,h,score]`（单类无 NMS），纯 INT8 会对整个输出张量用同一个量化尺度（被 0~640 的坐标通道主导），把 0~1 的 score 通道压成全 0 → 无检出。此时用 `--qtable qtable_f16.txt` 让 score 相关尾部算子保持 F16（其余仍 INT8），体积/速度接近纯 INT8 而精度正常。

## 精度与性能（实测，zhgd_without_nms_640 / BM1688）

在 tpu-mlir 1.27 容器内用 `model_runner`（CPU cmodel 模拟，单帧 1×3×640×640）对比 ONNX 参考输出：

| 模型 | 大小 | 输出 cosine 相似度 | max 绝对误差 | mean 相对误差 |
|------|------|------------------|-------------|--------------|
| F16  | 10.3 MB | 0.999999 | 1.54 | 0.25% |
| INT8（纯，置信度通道损坏） | 7.6 MB | 0.99979 | 98.9 | 置信度全 0 |
| INT8 + F16 混合（推荐） | 7.6 MB | 0.99982 | 101.6 | 置信度余弦 0.989 |

速度（cmodel CPU 模拟，仅相对参考；真实 BM1688 TPU 显著更快，INT8 相对 F16 通常快 3~5×）：

| 模型 | cmodel 平均耗时 |
|------|----------------|
| INT8 | ~6.5 s |
| F16  | ~35 s |

**BM1688 实机（demo_detection_sophgo，1×3×640×640，行人图，conf=0.5）**：

| 模型 | 推理耗时 | 检测框 |
|------|---------|--------|
| F16 | 15.7 ms | 3 框（score 0.83~0.87） |
| INT8 + F16 混合 | **3.7 ms** | 4 框（score 0.69~0.83，框位与 F16 基本一致） |

> INT8 混合精度相对 F16 在 BM1688 上**提速约 4.3×**。校准数据为训练集抽帧（100 张，kl 方法）。
