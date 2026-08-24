# ModelDeploy 文档系统重构 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 README 重构为概述门户，把细节文档下沉到 `docs/`，补全缺失能力文档，按语言拆分多语言 API 文档，并新增一键拉取模型脚本。

**Architecture:** 三层文档结构 —— README(概述门户) → docs/(细节库) → docs/README.md(总导航)。细节文档已大多存在(quickstart/backends/encryption/runtime_option/models)，本计划重点是：补新能力章节、拆分 apis.md、收敛内部笔记、把 README 中仍在维护的重复细节删除并指向 docs、新增 fetch 脚本与链接校验脚本。

**Tech Stack:** Markdown、PowerShell (Windows)、Bash (Linux/macOS)、CMake 无关。

## Global Constraints

- 全程中文文档内容；文档标题、导航、类名保留英文原样。
- 相对链接一律用仓库内相对路径（如 `./quickstart.md`、`../../README.md`），禁止外链占位。
- README 与 docs 内容不重复维护：同一信息只在一处，另一处链接。
- 不写任何未实现功能的文档（如 Qwen3）。
- 不删除任何已存在且仍有价值的文档；移动用 `git mv` 保留历史。
- 每个任务结束必须运行 `tools/check_docs_links.ps1`（Task 9 前用本机 ad-hoc 检查）验证无死链后方可 commit。

---
## 文件结构

**新建：**
- `tools/fetch_test_data.ps1` — Windows 一键下载 test_data.zip 并解压到仓库根 `test_data/`
- `tools/fetch_test_data.sh` — Linux/macOS 版
- `tools/check_docs_links.ps1` — 校验 docs/ 与 README 内所有相对 .md 链接目标存在
- `docs/conversion.md` — 模型转换/量化/混合精度/trtexec/bmodel 一站式
- `docs/api/README.md`、`docs/api/cpp.md`、`docs/api/python.md`、`docs/api/capi.md`、`docs/api/csharp.md`、`docs/api/rust.md`
- `docs/internal/`（含迁移来的 6 个内部笔记）

**修改：**
- `README.md` — 概述化、删除重复细节、补文档链接、清理空章节
- `docs/README.md` — 更新总导航
- `docs/quickstart.md` — 增加"一键拉取数据"小节（引用 fetch 脚本）
- `docs/models.md` — 补第 15~24 章新能力
- `docs/apis.md` — 内容拆分到 `docs/api/*` 后删除
- 迁移至 `docs/internal/`：`capi_bindings_analysis.md`、`capi_risk_register.md`、`capi_cpp_language_binding_best_practices.md`、`image_data_issue_register.md`、`optimization_prd.md`、`performance_analysis_2026.md`

---

### Task 1: 一键拉取模型/数据脚本

**Files:**
- Create: `tools/fetch_test_data.ps1`
- Create: `tools/fetch_test_data.sh`

**Interfaces:**
- Produces: `.ps1` 参数 `-Url <string>`、`-Force <switch>`、`-OutDir <string>`(默认仓库根)；`.sh` 同样语义 `--url`、`--force`、`--outdir`。退出码非 0 表示失败。

- [ ] **Step 1: 写 `tools/fetch_test_data.ps1`**

```powershell
#!/usr/bin/env pwsh
# 一键下载 ModelDeploy 测试数据(test_data.zip)并解压到仓库根 test_data/
[CmdletBinding()]
param(
    [string]$Url = "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/test_data.zip",
    [switch]$Force,
    [string]$OutDir = ""
)
$ErrorActionPreference = "Stop"

# 默认输出到本脚本上级目录(仓库根)
if ([string]::IsNullOrEmpty($OutDir)) {
    $OutDir = Split-Path -Parent $PSScriptRoot
}
$testDataDir = Join-Path $OutDir "test_data"
$zipPath = Join-Path $OutDir "test_data.zip"

if ((Test-Path -LiteralPath $testDataDir) -and -not $Force) {
    Write-Host "test_data/ 已存在(用 -Force 重新下载)。"
    exit 0
}

Write-Host "下载 $Url -> $zipPath"
try {
    Invoke-WebRequest -Uri $Url -OutFile $zipPath -UseBasicParsing
} catch {
    Write-Host "下载失败: $($_.Exception.Message)"
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    exit 1
}

Write-Host "解压 $zipPath -> $OutDir"
try {
    Expand-Archive -LiteralPath $zipPath -DestinationPath $OutDir -Force
} catch {
    Write-Host "解压失败: $($_.Exception.Message)"
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    exit 1
}
Remove-Item -LiteralPath $zipPath -Force
Write-Host "完成: $testDataDir"
```

- [ ] **Step 2: 写 `tools/fetch_test_data.sh`**

```bash
#!/usr/bin/env bash
# 一键下载 ModelDeploy 测试数据(test_data.zip)并解压到仓库根 test_data/
set -euo pipefail

URL="${URL:-https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/test_data.zip}"
FORCE=0
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

if [[ -z "$OUTDIR" ]]; then
  OUTDIR="$(cd "$(dirname "$0")/.." && pwd)"
fi
TEST_DATA_DIR="$OUTDIR/test_data"
ZIP="$OUTDIR/test_data.zip"

if [[ -d "$TEST_DATA_DIR" && "$FORCE" -eq 0 ]]; then
  echo "test_data/ 已存在(用 --force 重新下载)。"
  exit 0
fi

echo "下载 $URL -> $ZIP"
if ! curl -L -o "$ZIP" "$URL"; then
  echo "下载失败" >&2
  rm -f "$ZIP"
  exit 1
fi

echo "解压 $ZIP -> $OUTDIR"
if ! unzip -o -q "$ZIP" -d "$OUTDIR"; then
  echo "解压失败(需要 unzip)" >&2
  rm -f "$ZIP"
  exit 1
fi
rm -f "$ZIP"
echo "完成: $TEST_DATA_DIR"
```

- [ ] **Step 3: 验证 PowerShell 语法与帮助**

Run: `pwsh -NoProfile -Command "[System.Management.Automation.Language.Parser]::ParseFile('$PWD/tools/fetch_test_data.ps1', [ref]\$null, [ref]\$err) > \$null; if(\$err.Count -gt 0){ \$err | ForEach-Object { \$_.Message }; exit 1 } else { 'PS syntax OK' }"`
Expected: `PS syntax OK`

- [ ] **Step 4: 验证 idempotent 分支（不联网）**

Run: `New-Item -ItemType Directory -Force -Path "$PWD/tools/_tmp_testdata\test_data" | Out-Null; & "$PWD/tools/fetch_test_data.ps1" -OutDir "$PWD/tools/_tmp_testdata"`
Expected: `test_data/ 已存在(用 -Force 重新下载)。` 且退出码 0。随后 `Remove-Item -Recurse -Force "$PWD/tools/_tmp_testdata"`。

- [ ] **Step 5: 验证 bash 脚本语法**

Run: `bash -n "tools/fetch_test_data.sh"; echo "bash syntax exit=$?"`
Expected: `bash syntax exit=0`

- [ ] **Step 6: Commit**

```bash
git add tools/fetch_test_data.ps1 tools/fetch_test_data.sh
git commit -m "feat(tools): add one-click test data fetch scripts (ps1 + sh)"
```

---

### Task 2: 新增 `docs/conversion.md`（模型转换/量化）

**Files:**
- Create: `docs/conversion.md`

**Interfaces:**
- Consumes: `README.md` 第 3、4、5、6 章内容（将下沉）
- Produces: 标题锚点 `#3-onnxruntime混合精度`、`#4-动态量化减体积`、`#5-trt-engine生成`、`#6-bmodel生成`，供 README 链接，并链接 `./backends.md`、`./sophgo_cross_build_and_test.md`。

- [ ] **Step 1: 写 `docs/conversion.md`**（完整内容如下）

```markdown
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

基于 tpu-mlir 将 ONNX 转 `.bmodel`，工具见 [`tools/docker/sophgo/`](../../tools/docker/sophgo)。先准备 tpu-mlir 1.27 环境：

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
```

> 注意：上面缩进的 ` ``` ` 是文档代码块定界，实际写入时保持四级无缩进代码围栏。

- [ ] **Step 2: 校验相对链接目标存在**

Run: `git status --short 2>$null; foreach($l in @("docs/backends.md","docs/sophgo_cross_build_and_test.md","tools/docker/sophgo/README.md")){ if(!(Test-Path $l)){ Write-Output "BROKEN: $l" } } ; if(Test-Path "tools/docker/sophgo/README.md"){ 'sophgo README exists' }`
Expected: `sophgo README exists`（其余路径存在则无 BROKEN 输出）。若 `tools/docker/sophgo/README.md` 不存在，把 conversion 中该引用改为 `tools/docker/sophgo/`。

- [ ] **Step 3: Commit**

```bash
git add docs/conversion.md
git commit -m "docs: add conversion/quantization guide (docs/conversion.md)"
```

---

### Task 3: `docs/models.md` 补全新能力章节（15~24）

**Files:**
- Modify: `docs/models.md`（在文末第 14 章后追加第 15~24 章；并将第 14 章末句"各模型导出/转换教程见 [README](../../README.md)"改为"见 [模型转换与量化](./conversion.md) 与 [后端详解](./backends.md)"）

**Interfaces:**
- Consumes: `README.md` 第 7 节路线图标记；`docs/backends.md`、`docs/quickstart.md` 现有锚点。
- Produces: 第 15~24 章，每章含 `class` 名、最小用法、指向对应 `examples/` 示例文件的链接。

- [ ] **Step 1: 追加第 15~20 章**（跟踪/动作识别/文档理解/Re-ID/声纹/音频解决方案）

在 `docs/models.md` 末尾追加：

```markdown
## 15. 多目标跟踪（Tracking）

在检测结果基础上做跨帧目标跟踪，输出稳定 `track_id`。基础类 `BaseTracker`，实现 `ByteTracker` / `BoTSORT` / `StrongSORT`。

```cpp
// 头文件: #include "modeldeploy/tracking.h"
// 用法见示例: examples/demo_tracking/demo_tracking_ort_cpu.cpp
```

- `ByteTracker`：轻量、适合实时；`BaseTracker` 提供 `update(detections)` 返回带 track_id 的轨迹。
- 与 `UltralyticsDet` 配合：det → track → 可视化，跨帧保持稳定 `track_id`。
- **示例**：`examples/demo_tracking/`（`demo_tracking_ort_cpu.cpp` 演示 det→track→可视化整体流程）。

## 16. 视频动作识别（Video Action Recognition）

基于视频帧序列的动作分类。两类模型：

- **TSN**（RGB 帧，配合 `VideoDecoder` 抽帧）：`vision::action::TSN`
- **ST-GCN**（骨架，`UltralyticsPose` 提关键点后输入）：`vision::action::StGcn`

```cpp
// TSN: examples/demo_action/demo_action.cpp
// ST-GCN 骨架: examples/demo_action/demo_action_skeleton.cpp
```

- **示例**：`examples/demo_action/`（输入 mp4 视频，输出 top 动作 label+score）。

## 17. 文档理解（Document Understanding → Markdown）

版面分析 `StructureV2Layout` 定位版面/公式/表格，配合 OCR 与表格识别输出整页 Markdown：
公式以 `$...$`、表格以 HTML 呈现。

```cpp
// 完整管线见: examples/demo_doc/demo_doc.cpp
// 用法: demo_doc <layout.onnx> <image> [<formula.onnx> [dict]] [--ocr ...] [--table ...]
```

- **示例**：`examples/demo_doc/demo_doc.cpp`。

## 18. 行人 Re-ID（Person Re-Identification）

`vision::reid::ReID` 基于 OSNet 输出 512-d 行人特征，配合内存 `ReIdGallery` 做检索匹配。

```cpp
// 用法: examples/demo_reid/demo_reid.cpp <model> <imgA> <imgB>
```

- **示例**：`examples/demo_reid/demo_reid.cpp`（输出 embedding 维度 + gallery 匹配 label/score）。

## 19. 声纹验证（Speaker Verification）

ECAPA-TDNN 输出 192-d 说话人 embedding，配合内存 `SpeakerGallery` 验证/检索。

```cpp
// 纯音频，无需 OpenCV。用法: examples/demo_speaker/demo_speaker.cpp <model.onnx> <wavA> <wavB>
```

- **示例**：`examples/demo_speaker/demo_speaker.cpp`（输出两段语音 embedding 维度 + gallery 匹配 label/score）。

## 20. 音频解决方案（音频方案）

提供常用音频场景的预组装方案：

| 方案 | 头文件 | 说明 |
|------|--------|------|
| 说话人分段 `SpeakerDiarization` | `audio/solutions/speaker_diarization.h` | VAD 切段 |
| 说话人检索 `SpeakerSearch` | `audio/solutions/speaker_search.h` | 声纹检索 |
| 流式识别 `StreamingStt` | `audio/solutions/streaming_stt.h` | 分块 push + 回调 |
| TTS 批处理 `TtsBatcher` | `audio/solutions/tts_batcher.h` | enqueue/dequeue_all |

**示例**：`examples/demo_audio_solutions/demo_diarization.cpp`、`demo_stream_stt.cpp`、`demo_tts_batch.cpp`。
```

- [ ] **Step 2: 追加第 21~24 章**（NLP/条码/手部/解决方案）

```markdown
## 21. NLP（jieba 分词 / 分类）

基于 jieba 的中文处理工具与可选 BERT 文本分类（ONNX）。

```cpp
// 纯工具: Splitter / Keywords / Stats / Tokenizer / Normalizer
// 文本分类: nlp::TextClassifier("bert.onnx", option)
// 用法: examples/demo_nlp/demo_nlp.cpp [bert.onnx] [text]
```

**示例**：`examples/demo_nlp/demo_nlp.cpp`。

## 22. 条码 / 二维码（Barcode / QR）

`vision::barcode::BarcodeDetector` 纯 CV 识别（零 DNN，跨全部后端），输出格式、文本、分数与是否二维码。

```cpp
// 用法: examples/demo_barcode/demo_barcode.cpp
// 输出形如: [QR Code] https://example.com/MD (score, is_qr)
```

**示例**：`examples/demo_barcode/demo_barcode.cpp`。

## 23. 手部关键点 / 关键点扩展

- **手部关键点** `vision::hand::HandKeypoint`：检测手 + 关键点。示例 `examples/demo_hand/demo_hand.cpp`。
- **关键点扩展**：车辆关键点、面部 Landmark 106 点。示例 `examples/demo_landmark/demo_landmark.cpp`（`demo_landmark <vehicle.onnx|none> <face.onnx|none> <image.jpg>`）。

## 24. CV 解决方案（场景方案）

基于 `SolutionBase` 的预组装视觉场景方案，无需自带权重、多为纯算法：

| 方案 | 类 | 说明 |
|------|----|------|
| 跨线计数 | `ObjectCounter` | 统计 line_in/line_out 与类别计数 |
| 热力图 | `Heatmap` | 生成密度热力峰 |
| 测速 | `SpeedEstimator` | 估算移动速度 m/s |
| 车位管理 | `ParkingManager` | 车位占用判定 |

**示例**：`examples/demo_solutions/demo_solutions.cpp`；CV 纯工具（Annotator/LineZone/PolygonZone/Metrics mAP）见 `examples/demo_tools/demo_tools.cpp`。
```

> 上两步中缩进的代码围栏为 Markdown 定界，写入时保持无缩进。有关键字（如 ` ``` `）务必原样保留。写完用 Read 复核围栏成对。

- [ ] **Step 3: 修改第 14 章末句**

把 `docs/models.md` 第 277 行：
`各模型导出/转换教程见 [README](../../README.md) 与 [后端详解](./backends.md)。`
改为：
`各模型导出/转换教程见 [模型转换与量化](./conversion.md) 与 [后端详解](./backends.md)。`

- [ ] **Step 4: 复核 Markdown 围栏成对**

Run: `$c = Get-Content docs/models.md -Raw; $n = ([regex]::Matches($c, '```')).Count; "code fences: $n (应为偶数)"`
Expected: `code fences: N (应为偶数)`，其中 N 为偶数；若为奇数则修复围栏。

- [ ] **Step 5: Commit**

```bash
git add docs/models.md
git commit -m "docs: add new capability chapters (tracking/action/doc/reid/speaker/nlp/barcode/hand/solutions) to models.md"
```

---

### Task 4: 按语言拆分 `docs/apis.md` → `docs/api/`

**Files:**
- Create: `docs/api/README.md`、`docs/api/cpp.md`、`docs/api/python.md`、`docs/api/capi.md`、`docs/api/csharp.md`、`docs/api/rust.md`
- Delete: `docs/apis.md`

**Interfaces:**
- Consumes: `docs/apis.md` 全文（各语言小节内容）
- Produces: `api/README.md` 概览 + 每语言文档。路由：`docs/README.md` 改为指向 `./api/README.md`；`docs/quickstart.md`、`docs/models.md` 内的 `apis.md` 链接更新为 `./api/README.md`。

- [ ] **Step 1: 写 `docs/api/README.md`**

```markdown
# ModelDeploy 多语言 API

ModelDeploy 核心逻辑全部在 C++ SDK，提供 **C++ / Python / C / C# / Rust** 五种绑定，其余语言是对 C++/C 的薄封装，行为一致。

| 语言 | 文档 | 定位 |
|------|------|------|
| C++ | [cpp](./cpp.md) | 首选，完整功能全部模型/后端 |
| Python | [python](./python.md) | pybind11，科学计算场景 |
| C | [capi](./capi.md) | 嵌入式 / FFI 桥接（`md_*` 前缀） |
| C# | [csharp](./csharp.md) | .NET（`ModelDeploy` 命名空间） |
| Rust | [rust](./rust.md) | FFI 封装 C API |

后端/设备/精度配置在所有语言中保持一致，见 [RuntimeOption 配置](../runtime_option.md)。
```

- [ ] **Step 2: 拆出各语言文档**

把 `docs/apis.md` 的对应小节（保留正文与代码，补齐每篇的独立标题/引言/链接）写入各文件：
- `cpp.md` ← apis.md 第 1 节（链接到 `../quickstart.md#3-编写第一个检测程序`）
- `python.md` ← apis.md 第 2 节（2.1 安装 / 2.2 使用 / 2.3 已绑定模块 / 2.4 性能测试）
- `capi.md` ← apis.md 第 3 节（含"接口分组"表；编译需 `BUILD_CAPI=ON`）
- `csharp.md` ← apis.md 第 4 节；补充：运行示例 `cd csharp && dotnet build ModelDeployExample/ModelDeployExample.csproj -c Debug`，参考 [EXAMPLES-C#](../../examples/EXAMPLES.md)
- `rust.md` ← apis.md 第 5 节并**补覆盖率**：

```markdown
# Rust 绑定

Rust 通过 FFI 封装 C API。目录 `rust/modeldeploy/`：

```rust
use modeldeploy::runtime::RuntimeOption;

let mut option = RuntimeOption::new();
option.ort_backend();   // 或 sophgo_backend(0)/trt_backend()/mnn_backend()
```

## 主要模块文件

| 文件 | 说明 |
|------|------|
| `src/runtime.rs` | `RuntimeOption` 封装 |
| `src/ffi.rs` | FFI 声明（`MD_BACKEND_*` 常量等） |
| `src/model.rs` | 模型加载/推理 |
| `src/audio.rs` / `src/barcode.rs` / `src/nlp.rs` / `src/reid.rs` / `src/solution.rs` / `src/tracker.rs` | 各能力封装 |
| `src/image.rs` | 图像处理 |
| `src/types.rs` / `src/error.rs` | 类型与错误 |

## 示例（运行于 `rust/modeldeploy/examples/`）

内置 11 个示例：`classification` / `depth` / `detection` / `face_age` / `face_detection` / `face_gender` / `face_rec` / `obb` / `pose` / `seg` / `sem`。

```bash
cd rust/modeldeploy
cargo run --example detection
```
```

> 说明：`src/reid.rs` 若实际不存在则以 `src` 目录真实 `.rs` 文件为准调整上表（运行时用 `Get-ChildItem rust/modeldeploy/src/*.rs` 核对）。此表必须与实际文件一致。

- [ ] **Step 3: 校验 `rust/modeldeploy/src/*.rs` 实际文件，修正 rust.md 模块表**

Run: `Get-ChildItem rust/modeldeploy/src/*.rs | ForEach-Object { $_.Name }`
Expected: 输出实际文件列表；将 Step 2 的 rust.md 模块表修正为该列表与 `src` 中真实职责（用 grep 看各文件顶层结构确认职责描述），确保无缺漏/多余。

- [ ] **Step 4: 删除 `docs/apis.md` 并更新引用**

Run: `git rm docs/apis.md`；随后全局替换对 `apis.md` 的引用为 `api/README.md`：
`git grep -l "apis.md" -- docs README.md | ForEach-Object { (Get-Content $_ | ForEach-Object { $_ -replace '\./apis\.md','./api/README.md' -replace 'apis\.md','api/README.md' }) | Set-Content $_ }`
（仅替换指向该文档的链接文本，不改正文描述。）

- [ ] **Step 5: Commit**

```bash
git add docs/api docs/apis.md
git commit -m "docs: split apis.md into per-language docs/api/{cpp,python,capi,csharp,rust}.md"
```

---

### Task 5: 迁移内部笔记到 `docs/internal/`

**Files:**
- Move: `docs/capi_bindings_analysis.md`、`docs/capi_risk_register.md`、`docs/capi_cpp_language_binding_best_practices.md`、`docs/image_data_issue_register.md`、`docs/optimization_prd.md`、`docs/performance_analysis_2026.md` → `docs/internal/`
- Create: `docs/internal/README.md`（说明该目录为 SDK 开发内部笔记，非面向使用者的公开文档）

**Interfaces:**
- Produces: `docs/internal/` 目录及导航说明；这些文档从 `docs/README.md` 公开导航中移除。

- [ ] **Step 1: 创建 `docs/internal/README.md`**

```markdown
# ModelDeploy 内部笔记

本目录存放 **SDK 开发内部笔记**（风险登记、方案分析、性能分析、内部 PRD 等），**非面向最终使用者的公开文档**。公开文档见 [文档中心](../README.md)。

- `capi_risk_register.md` — capiv2 风险登记表
- `capi_bindings_analysis.md` — capiv2 C#/Rust 绑定对比分析
- `capi_cpp_language_binding_best_practices.md` — capi 稳定 C ABI 最佳实践
- `image_data_issue_register.md` — ImageData 问题登记簿
- `optimization_prd.md` — 20 路 25FPS 性能优化 PRD
- `performance_analysis_2026.md` — 全模型性能分析报告（2026-08）
```

- [ ] **Step 2: `git mv` 迁移文件**

Run: `git mv docs/capi_bindings_analysis.md docs/internal/ ; git mv docs/capi_risk_register.md docs/internal/ ; git mv docs/capi_cpp_language_binding_best_practices.md docs/internal/ ; git mv docs/image_data_issue_register.md docs/internal/ ; git mv docs/optimization_prd.md docs/internal/ ; git mv docs/performance_analysis_2026.md docs/internal/`
Expected: 无报错；`docs/internal/` 含 6 文件 + README。

- [ ] **Step 3: 移除 `docs/README.md` 中指向这些内部文件的链接（如有）**

Run: `git grep -n "capi_risk_register\|image_data_issue_register\|performance_analysis_2026\|optimization_prd\|capi_bindings_analysis\|capi_cpp_language_binding" -- docs/README.md`
Expected: 若有命中，手动从 `docs/README.md` 中删除对应行（这些属内部文档，不列入公开导航）。

- [ ] **Step 4: Commit**

```bash
git add docs/internal docs/README.md
git commit -m "docs: move internal dev notes to docs/internal/"
```

---

### Task 6: 将 README 瘦身为概述门户

**Files:**
- Modify: `README.md`（整篇重写为概述门户）

**Interfaces:**
- Consumes: Task 2 的 `docs/conversion.md` 锚点；已有 `docs/quickstart.md`、`docs/encryption.md`、`docs/backends.md`、`docs/models.md`、`docs/api/README.md`、`docs/README.md`。
- Produces: 各外部相对链接指向上述存在文档。

- [ ] **Step 1: 重写 `README.md`**（用下列结构替换现有内容；保留项目一句简介 + 功能亮点 + 最小构建/拉数据/示例 + 支持矩阵 + 路线图 + 链接区）

```markdown
# ModelDeploy

多后端推理 SDK（OnnxRuntime / TensorRT / MNN / Sophgo TPU），支持检测/分割/姿态/OBB/分类/人脸/OCR/车牌/行人属性/Re-ID/条码二维码/语音(ASR/TTS/VAD)/声纹/文档理解(→Markdown)/跟踪/视频动作识别/NLP 等模型与能力，并提供 C++ / Python / C / C# / Rust 五种绑定。一套代码统一调用四种后端。

> 完整文档见 **文档中心 [docs/README.md](./docs/README.md)**。

## 功能亮点

- **多后端统一 API**：`RuntimeOption` 一键切换 OnnxRuntime / TensorRT / MNN / Sophgo(算能 TPU)
- **AI 视觉**：检测/分割/姿态/OBB/分类/深度/语义分割/人脸/车牌/OCR/行人属性/Re-ID/手势/条码二维码
- **AI 音频**：ASR(SenseVoice)/TTS(Kokoro)/VAD/声纹验证/说话人分段
- **文档理解**：版面分析 + 公式/OCR/表格 → Markdown
- **视频**：解码 + 动作识别(TSN/ST-GCN) + 多目标跟踪
- **NLP**：jieba 分词/分句/关键词/统计 + BERT 文本分类
- **解决方案层**：对象计数/热力图/测速/车位管理/说话人检索/流式 STT/TTS 批处理
- **模型加密**：AES-256-CBC 防模型权重泄露
- **多语言绑定**：C++ / Python / C / C# / Rust

## 快速开始

```bash
git clone https://github.com/ChaoII/ModelDeploy.git && cd ModelDeploy
# Windows 用 "x64 Native Tools Command Prompt for VS 2022"; 推荐 Ninja
cmake -S . -B build -G Ninja -DBUILD_AUDIO=ON -DBUILD_VISION=ON \
      -DBUILD_CAPI=OFF -DBUILD_PYTHON=OFF -DENABLE_MNN=OFF \
      -DENABLE_ORT=ON -DENABLE_TRT=OFF -DWITH_GPU=OFF
cmake --build build --config Release --parallel
cmake --install build
```

安装后生成 `install/`（`include/` + `lib/`）。详细编译选项与第一个程序见 [快速开始](./docs/quickstart.md)。

### 拉取测试数据

```bash
# Windows
powershell -ExecutionPolicy Bypass -File tools/fetch_test_data.ps1
# Linux/macOS
bash tools/fetch_test_data.sh
```

### 最小检测示例

```cpp
#include "modeldeploy/vision.h"
int main() {
    modeldeploy::RuntimeOption option;
    option.use_ort_backend(); option.use_cpu();
    auto det = modeldeploy::vision::detection::UltralyticsDet("yolo11n.onnx", option);
    det.get_preprocessor().set_size({640, 640});
    auto img = modeldeploy::ImageData::imread("test.jpg");
    std::vector<modeldeploy::vision::DetectionResult> result;
    det.predict(img, &result);
    return 0;
}
```

## 支持矩阵

| 后端 | 格式 | CPU | CUDA | OpenCL | TPU |
|------|------|-----|------|--------|-----|
| OnnxRuntime | `.onnx` | ✅ | ✅ | ✅ | — |
| TensorRT | `.engine`/`.onnx` | — | ✅ | — | — |
| MNN | `.mnn` | ✅ | ✅ | ✅ | — |
| Sophgo | `.bmodel` | — | — | — | ✅ (BM1688/CV186X) |

## 路线图

- [x] 重构 `Tensor` 支持 CUDA
- [x] Python / C# 绑定
- [x] Pipeline DAG 编排、视频解码、多目标跟踪、动作识别、文档理解、Re-ID、声纹、解决方案层
- [x] 多后端（ORT/TRT/MNN/Sophgo）统一 API + 模型加密
- [ ] 更多 CUDA 预处理函数

## 更多文档

| 主题 | 文档 |
|------|------|
| 文档中心(总导航) | [docs/README.md](./docs/README.md) |
| 快速开始(构建/首个程序) | [docs/quickstart.md](./docs/quickstart.md) |
| 架构 | [docs/architecture.md](./docs/architecture.md) |
| 后端详解 | [docs/backends.md](./docs/backends.md) |
| 模型转换/量化 | [docs/conversion.md](./docs/conversion.md) |
| 模型详解 | [docs/models.md](./docs/models.md) |
| 预处理 | [docs/preprocess.md](./docs/preprocess.md) |
| 性能优化 | [docs/performance.md](./docs/performance.md) |
| 多语言 API | [docs/api/README.md](./docs/api/README.md) |
| 模型加密 | [docs/encryption.md](./docs/encryption.md) |
| 多线程 | [docs/multi_thread.md](./docs/multi_thread.md) |
| Sophgo TPU | [docs/sophgo_cross_build_and_test.md](./docs/sophgo_cross_build_and_test.md) |
| 示例 | [examples/EXAMPLES.md](./examples/EXAMPLES.md) |
```

> 说明：保留仓库原有的英文/中文混排风格；`# C、C++ 绑定` 等核对后按需保留。原 README 中被下沉的内容（加密细节、混合精度、量化、trtexec、bmodel、model_encrypted 用法、第 8 节空占位）不再出现在 README，统一指向 docs/。

- [ ] **Step 2: 校验所有相对链接目标存在**

Run: `foreach($t in @("docs/README.md","docs/quickstart.md","docs/architecture.md","docs/backends.md","docs/conversion.md","docs/models.md","docs/preprocess.md","docs/performance.md","docs/api/README.md","docs/encryption.md","docs/multi_thread.md","docs/sophgo_cross_build_and_test.md","examples/EXAMPLES.md","tools/fetch_test_data.ps1","tools/fetch_test_data.sh")){ if(!(Test-Path $t)){ Write-Output "BROKEN: $t" } }; "link target check done"`
Expected: `link target check done` 且无 BROKEN 输出。

- [ ] **Step 3: Commit**

```bash
git add README.md docs/conversion.md
git commit -m "docs: slim README to overview portal, delegate detail to docs/"
```

---

### Task 7: 更新 `docs/README.md` 总导航与 `docs/quickstart.md`

**Files:**
- Modify: `docs/README.md`（更新导航：指向 conversion.md、api/、internal 不在公开导航；模型配置位置）
- Modify: `docs/quickstart.md`（增加"拉取测试数据"小节）

**Interfaces:**
- Produces: `docs/README.md` 索引与最终目录一致；quickstart 引用 `tools/fetch_test_data.ps1/.sh`。

- [ ] **Step 1: 更新 `docs/README.md` 导航**

在"文档导航"中新增/更新表格：
- "进阶"区新增一行：`| [模型转换/量化](./conversion.md) | 混合精度、动态量化、TRT engine、Sophgo bmodel 转换 |`
- "多语言 API"链接由 `./apis.md` 改为 `./api/README.md`
- 确认不包含指向 `docs/internal/` 的用户链接

- [ ] **Step 2: 在 `docs/quickstart.md` 增加"拉取测试数据"小节**

在第 2 节"构建 SDK"之后、第 3 节"编写第一个检测程序"之前插入：

```markdown
## 2.5 拉取测试数据与模型

```bash
# Windows
powershell -ExecutionPolicy Bypass -File tools/fetch_test_data.ps1
# Linux/macOS
bash tools/fetch_test_data.sh
```

脚本从 modelscope 拉取 `test_data.zip` 并解压到仓库根 `test_data/`（含测试图片 `test_images/` 与测试模型 `test_models/`）。重新下载加 `-Force` / `--force`。
```

- [ ] **Step 3: 校验链接并 Commit**

Run: `git grep -n "apis.md" -- docs/ README.md`
Expected: 无命中（apis.md 已删除且引用已更新）。

```bash
git add docs/README.md docs/quickstart.md
git commit -m "docs: update doc center nav + quickstart fetch data section"
```

---

### Task 8: 链接校验脚本 `tools/check_docs_links.ps1`

**Files:**
- Create: `tools/check_docs_links.ps1`

**Interfaces:**
- Produces: 扫描 `docs/**/*.md` 与根 `README.md`、`examples/EXAMPLES.md`，校验每个相对 markdown 链接的目标存在；退出码 0=无死链，1=存在死链。

- [ ] **Step 1: 写 `tools/check_docs_links.ps1`**

```powershell
#!/usr/bin/env pwsh
# 校验 docs/ 与 README 内相对 .md 链接目标是否存在。
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$files = Get-ChildItem -Path (Join-Path $root "docs") -Recurse -Filter *.md
$files += Get-Item (Join-Path $root "README.md")
$files += Get-Item (Join-Path $root "examples/EXAMPLES.md")
$broken = 0
foreach ($f in $files) {
    $text = Get-Content -LiteralPath $f.FullName -Raw
    $matches = [regex]::Matches($text, '\]\(([^)#]+?)(?:#.*)?\)')
    foreach ($m in $matches) {
        $link = $m.Groups[1].Value.Trim()
        if ($link -match '^(https?://|mailto:)') { continue }
        if ($link -match '\.(png|jpg|jpeg|gif|svg)($|\s)') { continue }  # 图片链接忽略正文部分检查
        $target = Join-Path (Split-Path $f.FullName) $link
        if (-not (Test-Path -LiteralPath $target)) {
            Write-Output ("BROKEN: {0} -> {1} (in {2})" -f $link, $target, $f.FullName)
            $broken++
        }
    }
}
if ($broken -gt 0) { Write-Output "共 $broken 个死链"; exit 1 }
Write-Output "链接校验通过"
```

- [ ] **Step 2: 运行并修复全部死链**

Run: `pwsh -NoProfile -File "tools/check_docs_links.ps1"`
Expected: `链接校验通过`。若输出 BROKEN 行，逐个修正目标(多数因 Task 4 删除 apis.md / 重命名 reference 造成)。重复运行直至通过。
> 注：`image_data_issue_register.md` 等内部文档可能引用 test 数据相对路径，若误报可按图片链接例外或路径修正处理，不应删内容。

- [ ] **Step 3: Commit**

```bash
git add tools/check_docs_links.ps1
git commit -m "feat(tools): add docs link checker (check_docs_links.ps1)"
```

---

### Task 9: 全量复核与收尾

**Files:**
- （无新文件；仅复核）

**Interfaces:**
- Consumes: 前述所有任务的最终文档结构。

- [ ] **Step 1: 复核文档树与链接**

Run: `git ls-files | Select-String "docs/" | ForEach-Object { $_ } `；期望看到：`conversion.md`、`api/README.md` 与 `api/*.md`、`internal/` 六文件；`apis.md` 已删除。
Run: `pwsh -NoProfile -File "tools/check_docs_links.ps1"`
Expected: `链接校验通过`。

- [ ] **Step 2: 对照验收标准核对**

逐项确认：
1. `README.md` 为概述门户，无大段细节，含指向 docs/ 的链接。→ 满足(Task 6)
2. `docs/conversion.md`、`docs/api/`、`docs/internal/` 存在且完整。→ Task 2/4/5
3. `docs/models.md` 覆盖 15~24 章。→ Task 3
4. `docs/README.md` 索引与目录一致。→ Task 7
5. `tools/fetch_test_data.ps1` 与 `.sh` 可运行。→ Task 1
6. 全库相对链接校验通过。→ Task 8/9

- [ ] **Step 3: 报告完成**

输出：最终文档树、每个验收项勾选结果、`git log --oneline -12` 展示本次提交序列。
```

## Self-Review

**Spec coverage 对照：**
- README 概述化 → Task 6
- docs/conversion.md → Task 2
- backends.md 吸收 TRT engine → 已在 backends.md 5.4 存在，Task 6 让 README 指向它（+conversion.md 汇总）
- runtime_option.md 补模型配置(8.1/8.2/8.3) → spec 曾包含，但 runtime_option.md 已有第 8 节各后端 Option 结构即为该配置；确认快速核对，若缺"通用/输入/输出配置"则本计划以 runtime_option 现有第 8 节为准，不重复造（**决定：视为已覆盖，见下方说明**）
- models.md 新能力 → Task 3
- api 拆分 → Task 4
- internal 迁移 → Task 5
- fetch 脚本 → Task 1
- 链接校验 → Task 8/9

**Placeholder scan：** 无 TBD/TODO；所有脚本与新增文档正文完整给出；rust.md 模块表"以实际文件为准"是明确的核对动作而非占位。

**Type consistency：** fetch 脚本参数名 `.ps1`(-Url/-Force/-OutDir) 与 `.sh`(--url/--force/--outdir) 在各自引用处一致；`conversion.md` 锚点与 README 链接一致；`api/README.md` 与各子文档命名一致。

> **runtime_option 说明**：spec 中的"补模型配置 8.1/8.2/8.3"经核对 `docs/runtime_option.md` 第 8 节"各后端 Option 结构"已系统覆盖 `ort_option/mnn_option/trt_option/sophgo_option`，即通用配置等已在 docs 落地，故 README 空占位删除、内容由 runtime_option.md 承接即满足 spec，无需新增重复章节。

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-24-docs-system-refactor.md`.
