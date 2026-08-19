# 设计文档：测试/基准迁移到 yolo26n 家族 + ppocrv6_tiny

日期：2026-08-19
分支：capi-v2

## 背景

demo 系列已统一到 yolo26n 家族（det/seg/pose/obb/cls/sem/depth）与 OCR ppocrv6_tiny，
分类输入 224。但单元测试与回归基线（`tests/baselines/`，已提交 44 个 JSON）仍指向旧模型
（yolo11n 家族 + ppocrv4）。本迁移把测试与基准对齐到新模型。

磁盘上已存在完整新模型资产：
- `test_data/test_models/{onnx,mnn,trt,sophgo}/yolo26n/*`（含 TRT 动态批 `yolo26n_b8.engine`）
- `test_data/test_models/onnx/ocr/ppocrv6_tiny/{det,cls,rec}_infer.onnx`、`mnn/ocr/ppocrv6_tiny/*`
- `ppocrv6_tiny_dict.txt`

## 目标

把下列测试套件从 yolo11n/ppocrv4 迁到 yolo26n/ppocrv6_tiny，重生成基线，并在
本地（ort/mnn/trt）与 sophgo 设备（.70）上验证通过。

## 关键决策（已与用户确认）

1. **全部迁移**：`baseline_compare`、`test_vision_models`、`test_capi`、`test_pipelines` 全部迁移。
2. **删变体**：yolo26n 自带 end2end NMS，缺 `_nms`/`_without_nms`/pre-raw 变体；
   删除这些变体专属回归用例，保留端到端 det/seg/pose/obb/cls 回归。
3. **sophgo**：沿用 int8/f16 量化；命名标准化为「短横线 + 小写量化类型」
   （`yolo26n-seg-int8.bmodel` / `yolo26n-seg-f16.bmodel`）。需在磁盘上把现有
   `_F16`/`_INT8` 文件物理改名为 `-f16`/`-int8`，并同步更新 demo 与测试的所有引用。
   流程按 `docs/sophgo_cross_build_and_test.md` 执行（.243 交叉编译 → .70 设备跑
   baseline_collect 重生成基线）。

## 改动清单

| 文件 | 改动 |
|---|---|
| `tests/baseline_compare.cpp` | `yolo11n/*`→`yolo26n/*`；OCRv4→v6_tiny；删 `_nms`/pre-raw 变体用例；sophgo 同理 |
| `tests/test_vision_models.cpp` | `yolo11n/*`→`yolo26n/*`；OCRv4→v6_tiny（dict=`ppocrv6_tiny_dict.txt`）；行为断言不变 |
| `tests/test_capi.cpp` | `yolo11n/*`→`yolo26n/*`；`ppocrv4_mobile`→`ppocrv6_tiny`；`ppocrv4_dict.txt`→`ppocrv6_tiny_dict.txt` |
| `tests/test_pipelines.cpp` | `ocr_dict()` 增 `ppocrv6_tiny_dict.txt`（优先 v6_tiny） |
| `tests/baselines/{ort,mnn,trt,sophgo}/*.json` | 删除旧 yolo11n/ppocrv4 基线，替换为 yolo26n/ppocrv6_tiny 新基线 |
| demo sophgo 路径 | `_F16`/`_INT8` → `-f16`/`-int8` 命名（含 `examples/demo_*` sophgo 文件与生成器） |

## 基线重生成方法

1. 本地 ORT 为基准：`tests/baseline_collect.cpp` 具备 `collect_yolo/cls/ocr_det/ocr_rec/ocr_cls`
   能力，对 yolo26n/ppocrv6_tiny 在 ORT 上收集 `tests/baselines/ort/*.json`。
2. mnn/trt：以 ORT 基线为参照（`warn_diff`），并生成 self 基线（`require_no_diff`）。
3. sophgo：在 .70 设备上跑 `baseline_collect` 生成 `tests/baselines/sophgo/*.json`（int8/f16，
   `-int8`/`-f16` 命名）。

## 验证

- 本地：`./test_modeldeploy [regression]`、`[vision_models]`、`[pipeline]`、`[capi]`；
  GPU 相关（`[gpu]`）单跑 ort_gpu/trt/mnn_cuda。
- sophgo：设备上跑对应 `[backend:sophgo]` tag。
- 全量 `ctest`。

## 非目标

- 不改 SDK 推理逻辑（不影响正确性测试语义）。
- 不新增模型转换脚本（复用现有 trtexec / mnnconvert / sophgo 转换流程）。
- 保留行为断言风格（`size>0`、label>=0 等），不引入硬编码期望值。
