# SOPHGO bmodel 产物与清理记录

文档日期: 2026-09-01
涉及目录: `test_data/test_models/sophgo/`(gitignore, 测试数据)
对应工具: `tools/convert/`(models.json 注册表 + convert_bmodel.ps1 + conv_in_docker.sh)

## 1. 目录原则

`test_models/sophgo/` **只存放最终可部署的 bmodel**。tpu-mlir 转换的中间产物
(`*.mlir`、`*_top_f32_all_weight.npz`、`*_tpu_addressed_*_weight.npz`、
`*_cali_table.txt`、`*_data_list.txt`、`*_opt.onnx.prototxt`、
`*_bm1688_*.layer_group_*.json`、`*.ref_files.json`、`*-int8.bmodel.json`、
`*-int8.bmodel.net_0.profile`、`compiler_profile_*.txt`、`_cali_*` 校准目录、
`*-int8/` / `*-f16*/` 编译调试子目录 等)均为**可再生**产物,不再入库目录。

清理口径(引用/注册表驱动):
- **保留**: `tools/convert/models.json` 注册表内全部模型(23/25, 2 个 skip 见 §5) +
  代码/示例实际引用的 bmodel + `yolo11n/` 整目录(demo 默认模型) + `.gitkeep`。
- **备份**(移出目录, 见 §3): 未被任何代码引用的历史/实验 bmodel。
- **删除**: 可再生 tpu-mlir 中间产物(清单备份在 §3 的 manifest)。

## 2. 保留产物(47 文件 / 约 525 MB)

### 规范注册表产物(models.json 驱动, verify_precision 校验)

| key | 文件 | 量化 | shape | cmodel cos |
|---|---|---|---|---|
| yolo26n | yolo26n/yolo26n-int8.bmodel | INT8+qtable(解码头F16) | 1x3x640x640 | 0.9996 |
| yolo26n-cls | yolo26n/yolo26n-cls-int8.bmodel | INT8 | 1x3x224x224 | 0.8199 |
| yolo26n-obb | yolo26n/yolo26n-obb-int8.bmodel | INT8 | 1x3x1024x1024 | 0.5993 |
| yolo26n-pose | yolo26n/yolo26n-pose-int8.bmodel | INT8 | 1x3x640x640 | 0.9995 |
| yolo26n-seg | yolo26n/yolo26n-seg-int8.bmodel | INT8+qtable | 1x3x640x640 | 0.9919 |
| yolo26n-sem | yolo26n/yolo26n-sem-int8.bmodel | INT8 | 1x3x640x640 | 0.9943 |
| yolo26n-depth | yolo26n/yolo26n-depth-int8.bmodel | INT8 | 1x3x640x640 | 0.9964 |
| ocr-det | ocr/ppocrv6_tiny/det_infer-int8.bmodel | INT8 | 1x3x960x960 | 0.9874 |
| ocr-cls | ocr/ppocrv6_tiny/cls_infer-int8.bmodel | INT8 | 1x3x48x192 | 0.9997 |
| ocr-rec | ocr/ppocrv6_tiny/rec_infer-int8.bmodel | INT8 | 1x3x48x320 | 0.9817 |
| lpr-det | yolov5plate-int8.bmodel | INT8 | 1x3x640x640 | 0.9999 |
| lpr-rec | plate_recognition_color-int8.bmodel | INT8 | 1x3x48x168 | 0.9982 |
| scrfd | seetaface/scrfd_2.5g_bnkps_shape640x640-int8.bmodel | INT8 | 1x3x640x640 | 0.9921 |
| age | seetaface/age_predictor-int8.bmodel | INT8 | 1x3x256x256 | 0.9975 |
| gender | seetaface/gender_predictor-int8.bmodel | INT8 | 1x3x112x112 | 0.9999 |
| facerec | seetaface/face_recognizer-int8.bmodel | INT8 | 1x3x248x248 | 0.9385 |
| fas1 | seetaface/fas_first-int8.bmodel | INT8 | 1x3x224x224 | 1.0000 |
| det10g | insightface/buffalo_l/det_10g-int8.bmodel | INT8 | 1x3x640x640 | 0.9922 |
| 2d106 | insightface/buffalo_l/2d106det-int8.bmodel | INT8 | 1x3x192x192 | 0.9427 |
| 1k3d | insightface/buffalo_l/1k3d68-f16.bmodel | **F16** | 1x3x192x192 | 0.9992 |
| w600k | insightface/buffalo_l/w600k_r50-int8.bmodel | INT8 | 1x3x112x112 | 0.8284 |
| genderage | insightface/buffalo_l/genderage-int8.bmodel | INT8 | 1x3x96x96 | 0.9987 |
| zhgd-ml | zhgd_ml-int8.bmodel | INT8 | 1x3x1280x1280 | 0.9672 |

完整精度表见 `tools/convert/bmodel_precision_report.md`。

### 代码引用的历史 bmodel(保留, 勿删)

- `yolo26n/` 全 7 任务的 `-f16.bmodel` 变体(sophgo baseline/基准引用)。
- `zhgd_ml_int8.bmodel`、`zhgd_without_nms_640_int8.bmodel`(examples 下
  sophgo demo 默认模型)。
- `yolo11n/` 整目录(含 `*_int8.bmodel` / `*_f16.bmodel` / 无后缀 `.bmodel`,
  为 demo_classification/seg/pose/obb_sophgo 默认路径)。

## 3. 清理执行摘要

- **删除 383 个中间产物, 释放约 1.13 GB**。完整清单:
  `C:\Users\aichao\AppData\Local\Temp\opencode\sophgo_cleanup_backup\_deleted_intermediates_manifest.txt`
- **备份 15 个未引用 legacy bmodel, 约 132 MB** 至:
  `C:\Users\aichao\AppData\Local\Temp\opencode\sophgo_cleanup_backup\`
  (映射清单 `_moved_legacy_bmodels.txt`)。均为历史/实验/旧命名产物:
  - yolo26n 无后缀 `.bmodel`(det/cls/depth/obb/pose/seg/sem)
  - `yolo26n-det-f16.bmodel`(det F16; 代码引用的 `yolo26n-f16.bmodel` 本就不存在,
    sophgo det-f16 baseline 用例会自动 skip)
  - `yolo26n-{det,pose,seg}-int8-c3full.bmodel`、`yolo26n-f16-test.bmodel`
  - `zhgd_ml_f16.bmodel`、`zhgd_without_nms_{640,1280}.bmodel`
- 若需找回,直接按映射复制回原路径即可。

## 4. 中间产物再生方式

任何被删中间产物均可通过规范流水线再生成:

```bash
# 单个模型(容器内 tpu-mlir; 缺失产物自动重建)
powershell -File tools/convert/convert_bmodel.ps1 -Key yolo26n
# 全量(+qtable / F16 自动按 models.json 处理)
docker run --rm -v <test_models>:/conv -v <test_images>:/cali_img \
    -v tools/convert:/tconf -v tools/docker/sophgo:/tc \
    tpuc_dev:1.27-slim bash /tconf/conv_in_docker.sh
# 精度复验
python tools/convert/verify_precision.py
```

- 解码头 qtable: `tools/convert/yolo26n.qtable` / `yolo26n-seg.qtable`
  (`make_qtable.py` 可从 *_origin.mlir 按行/正则自动生成)。
- F16 支持: registry `quantize:"F16"` → 产出 `{stem}-f16.bmodel`。

## 5. 备注(为什么这样定)

- **yolo26n / yolo26n-seg**: 曾 int8 直接塌(输出坐标被压到 ±44, cos≈0.77)。
  根因是"坐标+分数"混合量纲单张量 int8 量化。修复=解码头 qtable(F16 输出链 +
  int8 backbone) → cos 0.9996 / 0.9919。
- **1k3d(landmark)**: int8 深度塌(尾 30%/60% F16 均无效, 全 F16 cos=0.9992 证明
  为早/中层塌缩) → 单走 **F16**(registry `quantize:"F16"`)。
- **yolo26n-obb**: 曾发现磁盘产物实为 640(registry 是 1024)导致 cmodel
  `overflow>=0` 断言 → 按 registry 重建 1024 后消除。obb int8 的 qtable 实测
  反而把坐标压到 155, 故保持纯 INT8(坐标保真, cos 0.6 可接受, 已在 registry note)。
- **fas2(已修复)**: 源为 TF-SSD 导出, 输入声明全动态但 BoxPredictor 的 Reshape
  目标却是按训练时单一分辨率烘焙的常量([1,1083,3,1] … 合计 1917 锚), 且背骨
  MobileNetV2 的 depthwise s2 Conv 在输入 <~300 时特征图缩成 1×1 直接报错,
  故任意尺寸都无法跑 —— 导出即 bug。由
  `tools/convert/repair_tf_ssd_reshape.py` 把 12 个 Reshape 锚数维改动态(-1),
  头部随分辨率自适应: ≤224 仍因背骨 Conv 不可用, **≥300 可用,
  300×300 时锚数恰还原 1917**。SDK `SeetaFaceAsSecond::size_` 默认 {300,300}
  与原生分辨率吻合, 无需改动。registry fas2 shape 改为 `1x3x300x300` 并去
  skip_reason。对应 MNN/TRT 已重转通过(见 bmodel_precision_report.md 下说明)。
- **zhgd-det 不可转(skip)**: 内建 NMS/GatherND 不被 tpu-mlir 1.27 支持,
  理由保留在 models.json `skip_reason` / `note`。
- **seetaface**: 5 个模型曾缺 `kernel_shape` 属性(tpu-mlir KeyError), 由
  `tools/convert/repair_conv_kernel_shape.py` 补齐后再转; age 输入应为 256(gemm 尺寸)。
- **命名约定**: 规范转换用 `{stem}-int8.bmodel` / `{stem}-f16.bmodel`(连字符);
  insightface 相关旧测试/基准用的是 `_f16.bmodel`(下划线), 仅在 Linux+Sophon-Sail
  (ENABLE_SOPHGO) 下运行, 与当前产物命名不一致, 属已知遗留, 不影响本机。
