# 跨后端回归基线测试系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 全覆盖迁移视觉模型到按后端分目录，跨后端（ORT/MNN/TRT/SOPHGO）回归基线测试，严格阈值验证量化精度差异。

**Architecture:** 模型文件迁移到 `test_data/test_models/{onnx,mnn,trt,sophgo}/`，基线迁到 `tests/baselines/{ort,mnn,trt,sophgo}/`。`baseline_collect` 加 `--backend` 路由，`baseline_compare` 每个"模型×后端"一个 TEST_CASE，同时做自对比（同后端基线）和基准对比（ORT 基线，严格阈值）。现有 ORT 测试从 `test_models/onnx/` 加载。

**Tech Stack:** C++17、nlohmann/json、Catch2 v3、ModelDeploySDK（ORT/MNN/TRT/SOPHGO）、trtexec、MNNConvert、tpu-mlir

## Global Constraints

- 模型目录：`test_data/test_models/{onnx,mnn,trt,sophgo}/`，onnx/ 下保留 face/、ocr/ 子目录结构
- 基线目录：`tests/baselines/{ort,mnn,trt,sophgo}/`，文件名 `<模型文件名>.<type>.json`
- 后端路由：onnx→ORT, mnn→MNN, engine→TRT, bmodel→SOPHGO（复用 `RuntimeOption::set_model_path` 自动路由）
- 对比阈值（严格，跨后端与自对比相同）：坐标±1px、Obb angle±0.5°、score±0.01、label严格、Seg mask nonzero_ratio差异<0.001、OCR text严格、Tensor数值±1e-4、Tensor shape严格
- 音频模型（sense_voice/、kokoro_v1_1/）不迁移
- 测试源文件需手动加入 `tests/CMakeLists.txt` 的 TEST_SOURCES（显式列表）
- 模型/基线/图片缺失 → `return` skip；文件存在但加载失败 → FAIL（不静默）
- 全覆盖迁移所有视觉模型 onnx 到 `onnx/`；yolo 系优先多后端转换
- 当前在 `main` 分支，工作区需干净

---

### Task 1: 目录创建 + 模型文件全覆盖迁移

**Files:**
- Modify: `test_data/test_models/`（大量文件移动，用 git mv 保留历史）

**Interfaces:**
- Produces: 目录结构 `test_models/{onnx,mnn,trt,sophgo}/`，onnx/ 含全部视觉模型（含 face/、ocr/ 子目录），mnn/trt/ 含已有 mnn/engine 模型，杂项归类

- [ ] **Step 1: 创建目录**

```powershell
cd E:\CLionProjects\ModelDeploy
New-Item -ItemType Directory -Force -Path test_data/test_models/onnx, test_data/test_models/mnn, test_data/test_models/trt, test_data/test_models/sophgo, test_data/test_models/onnx/face, test_data/test_models/onnx/ocr
New-Item -ItemType Directory -Force -Path tests/baselines/ort, tests/baselines/mnn, tests/baselines/trt, tests/baselines/sophgo
```

- [ ] **Step 2: 迁移根目录 onnx 到 onnx/**

```powershell
cd E:\CLionProjects\ModelDeploy
# 所有根目录 .onnx 移入 onnx/
Get-ChildItem test_data/test_models/*.onnx | ForEach-Object {
    git mv $_.FullName test_data/test_models/onnx/
}
# 列出确认
Get-ChildItem test_data/test_models/onnx/*.onnx | Select-Object -ExpandProperty Name
```
Expected: 根目录 onnx 全部移入，包含 yolo11n*.onnx、yolov5plate.onnx、plate_recognition_color.onnx、zhgd*.onnx、zc.onnx、best.onnx、line_edit.onnx、model.onnx

- [ ] **Step 3: 迁移 face/ 和 ocr/ 结构模型到 onnx/**

```powershell
cd E:\CLionProjects\ModelDeploy
# face 模型（非 onnx 辅助文件如无则跳过）
Get-ChildItem test_data/test_models/face/*.onnx | ForEach-Object {
    git mv $_.FullName test_data/test_models/onnx/face/
}
# ocr 结构模型（SLANet 等）
Get-ChildItem test_data/test_models/ocr/*.onnx | ForEach-Object {
    git mv $_.FullName test_data/test_models/onnx/ocr/
}
# 确认 face/ 和 ocr/ 子目录（非模型文件留在原位）
Get-ChildItem test_data/test_models/face -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
Get-ChildItem test_data/test_models/ocr -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
```
注意：face/ 和 ocr/ 下如有非 onnx 文件（.txt、.mnn 等），只移动 .onnx；ocr/ 下如全是 onnx 则整个目录移。迁移后如果 face/、ocr/ 原目录为空则 git mv 整个目录。

- [ ] **Step 4: 迁移已有 mnn/engine 到对应目录**

```powershell
cd E:\CLionProjects\ModelDeploy
# 根目录 mnn
Get-ChildItem test_data/test_models/*.mnn | ForEach-Object { git mv $_.FullName test_data/test_models/mnn/ }
# 根目录 engine：yolo 系带 .onnx.engine 双重后缀的重命名为单后缀
Get-ChildItem test_data/test_models/*.engine | ForEach-Object {
    $new = $_.Name -replace '\.onnx\.engine$', '.engine'
    git mv $_.FullName "test_data/test_models/trt/$new"
}
# 确认 trt/ 内容
Get-ChildItem test_data/test_models/trt | Select-Object -ExpandProperty Name
```
注意：`yolo11n_nms.onnx.engine`→`yolo11n_nms.engine`、`yolo11n-seg_nms.onnx.engine`→`yolo11n-seg_nms.engine`；zhgd 的 engine（`zhgd_det.engine` 等无双重后缀）保留原名。`model.engine` 在 Step 5 处理。

- [ ] **Step 5: 杂项归类（加密模型、mdenc）**

```powershell
cd E:\CLionProjects\ModelDeploy
# 加密 mdenc 模型归类（保持用途，移到 misc/ 或保留原处）
New-Item -ItemType Directory -Force -Path test_data/test_models/misc
# yolo11n_nms_encrypted.mdenc、model.mdenc 移入 misc/
Get-ChildItem test_data/test_models/*.mdenc -ErrorAction SilentlyContinue | ForEach-Object {
    git mv $_.FullName test_data/test_models/misc/
}
# model.engine 移入 trt/（保留原名 model.engine）
if (Test-Path test_data/test_models/model.engine) { git mv test_data/test_models/model.engine test_data/test_models/trt/ }
```

- [ ] **Step 6: 迁移现有基线到 ort/**

```powershell
cd E:\CLionProjects\ModelDeploy
Get-ChildItem tests/baselines/*.json | ForEach-Object { git mv $_.FullName tests/baselines/ort/ }
Get-ChildItem tests/baselines/ort/*.json | Select-Object -ExpandProperty Name
```
Expected: 15 个基线文件在 tests/baselines/ort/ 下

- [ ] **Step 7: 提交**

```bash
git add -A
git commit -m "refactor(test_data): 模型文件按后端分目录全覆盖迁移

所有视觉模型 onnx 移入 test_models/onnx/（含 face/ocr 子目录），
已有 mnn/engine 移入对应目录，杂项归类 misc/，
基线移入 tests/baselines/ort/。音频模型保持原位。"
```

- [ ] **Step 8: 验证构建未破坏**

```powershell
cd E:\CLionProjects\ModelDeploy\build
ctest --output-on-failure 2>&1 | Select-Object -Last 5
```
注意：此步骤会 FAIL（对比器还引用旧路径 `test_models/yolo11n.onnx`）。这是**预期的中间状态**——Task 2 修复对比器路径后通过。确认失败原因是"模型文件找不到"（skip），而非编译错误。

---

### Task 2: baseline_compare 支持 backend 目录路由

**Files:**
- Modify: `tests/baseline_compare.cpp`

**Interfaces:**
- Consumes: Task 1 的目录结构
- Produces: `model_path(rel, backend)` 返回 `test_models/<backend>/<rel>`；`baseline_dir(backend)` 返回 `tests/baselines/<backend>`；现有 11 个 TEST_CASE 改用 `onnx/` 路径；新增 ORT 全覆盖的额外 TEST_CASE

- [ ] **Step 1: 修改路径辅助函数**

```cpp
static fs::path get_test_data() {
    const char* env = std::getenv("TEST_DATA_DIR");
    if (env && *env) return fs::path(env) / "test_data";
    return fs::current_path() / "test_data";
}
static fs::path baseline_root() { return get_test_data().parent_path() / "tests" / "baselines"; }
static fs::path baseline_dir(const std::string& backend) {
    return baseline_root() / backend;
}
static fs::path model_path(const std::string& rel, const std::string& backend = "onnx") {
    return get_test_data() / "test_models" / backend / rel;
}
static fs::path image_path(const std::string& name) {
    return get_test_data() / "test_images" / name;
}
```

- [ ] **Step 2: 更新现有 TEST_CASE 的路径**

每个现有 TEST_CASE 改：
```cpp
// 旧: auto modelfile = model_path("yolo11n.onnx");
auto modelfile = model_path("yolo11n.onnx", "onnx");          // ORT 用 onnx/
auto base_file = baseline_dir("ort") / "yolo11n.onnx.det.json";  // 基线在 ort/
```
对全部 11 个现有 TEST_CASE 应用。注意 ocr 模型路径：
```cpp
auto modelfile = model_path("ocr/ppocrv4_mobile/det_infer.onnx", "onnx");
```
face 模型：
```cpp
auto modelfile = model_path("face/scrfd_2.5g_bnkps_shape640x640.onnx", "onnx");
```

- [ ] **Step 3: 编译 + 跑 [regression] 验证**

```powershell
cd E:\CLionProjects\ModelDeploy
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && cd /d E:\CLionProjects\ModelDeploy && cmake --build build --target test_modeldeploy"
cd build\bin
$env:PATH = "E:\CLionProjects\ModelDeploy\build\bin;" + $env:PATH
$env:TEST_DATA_DIR = "E:\CLionProjects\ModelDeploy"
.\test_modeldeploy.exe "[regression]"
```
Expected: 11 test cases / 43 assertions 全过（基线已迁到 ort/）

- [ ] **Step 4: 提交**

```bash
git add tests/baseline_compare.cpp
git commit -m "test: baseline_compare 支持按后端分目录路由（onnx→ort/基线）"
```

---

### Task 3: baseline_collect 支持 --backend 参数

**Files:**
- Modify: `tests/baseline_collect.cpp`

**Interfaces:**
- Consumes: Task 1 目录结构
- Produces: `--backend <ort|mnn|trt|sophgo>` 参数，路由 `test_models/<backend>/` + `baselines/<backend>/`

- [ ] **Step 1: 增加 backend 参数**

在 `Args` 结构加 `std::string backend = "ort";`，parse_args 加：
```cpp
else if (k == "--backend" && i + 1 < argc) a.backend = argv[++i];
```
usage 文本更新：
```cpp
<< " [--backend <ort|mnn|trt|sophgo>]"
```

- [ ] **Step 2: backend 到扩展名的映射**

```cpp
static std::string backend_to_ext(const std::string& backend) {
    if (backend == "mnn") return ".mnn";
    if (backend == "trt") return ".engine";
    if (backend == "sophgo") return ".bmodel";
    return ".onnx";  // ort 默认
}
```

- [ ] **Step 3: 路由模型路径和基线输出**

在 main 中，`--model` 若给的是相对文件名（如 `yolo11n.onnx`）则拼接 `test_models/<backend>/`；`--out` 若没给则默认 `tests/baselines/<backend>/`。具体：在解析后、collect 前：
```cpp
// 若 --model 不是绝对路径且不含 test_models，尝试按 backend 目录补全
if (!fs::exists(args.model)) {
    fs::path guess = fs::path("test_data") / "test_models" / args.backend / args.model;
    if (fs::exists(guess)) args.model = guess.string();
}
```

- [ ] **Step 4: 编译 + 冒烟测试**

```powershell
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && cd /d E:\CLionProjects\ModelDeploy && cmake --build build --target baseline_collect"
cd E:\CLionProjects\ModelDeploy\build\bin
$env:PATH = "E:\CLionProjects\ModelDeploy\build\bin;" + $env:PATH
.\baseline_collect.exe --model yolo11n.onnx --image E:\CLionProjects\ModelDeploy\test_data\test_images\test_detection0.jpg --out E:\CLionProjects\ModelDeploy\tests\baselines\ort --backend ort --type det
```
Expected: 生成 `tests/baselines/ort/yolo11n.onnx.det.json`

- [ ] **Step 5: 提交**

```bash
git add tests/baseline_collect.cpp
git commit -m "feat: baseline_collect 支持 --backend 参数路由后端目录"
```

---

### Task 4: 新增跨后端 TEST_CASE（mnn/trt）

**Files:**
- Modify: `tests/baseline_compare.cpp`

**Interfaces:**
- Consumes: Task 2 的 `model_path(rel, backend)` / `baseline_dir(backend)`
- Produces: 每"模型×后端"一个 TEST_CASE，标签 `[regression]` + `[backend:mnn]`/`[backend:trt]`，同时自对比 + 与 ORT 基准对比

- [ ] **Step 1: 写 MNN 测试模板（以 yolo11n 为例）**

```cpp
TEST_CASE("Regression: yolo11n detection MNN", "[regression][backend:mnn]") {
    auto modelfile = model_path("yolo11n.mnn", "mnn");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;
    auto ort_file = baseline_dir("ort") / "yolo11n.onnx.det.json";
    auto self_file = baseline_dir("mnn") / "yolo11n.mnn.det.json";
    if (!fs::exists(ort_file)) return;

    UltralyticsDet model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());
    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));

    // 自对比：同后端基线（存在则比）
    if (fs::exists(self_file)) {
        require_no_diff(compare_detection(load_json(self_file)["results"], results));
    }
    // 基准对比：与 ORT 基线（严格阈值）
    require_no_diff(compare_detection(load_json(ort_file)["results"], results));
}
```

- [ ] **Step 2: 为 mnn 后端添加 yolo11n、yolo11n_nms 测试**

（yolo11n.mnn 和 yolo11n_nms.mnn 已存在。seg/obb/pose/cls 的 mnn 需 Task 6 转换后才生成基线，测试先写好，缺失自动 skip。）

对 yolo11n_nms：
```cpp
TEST_CASE("Regression: yolo11n_nms detection MNN", "[regression][backend:mnn]") {
    // model_path("yolo11n_nms.mnn", "mnn"), image test_detection0.jpg
    // self baseline: mnn/yolo11n_nms.mnn.det.json, ort baseline: ort/yolo11n_nms.onnx.det.json
    // 与上面模板相同结构
}
```

- [ ] **Step 3: 为 trt 后端添加 yolo11n、yolo11n_nms、yolo11n-seg_nms 测试**

已有 engine：yolo11n.engine、yolo11n_nms.engine、yolo11n-seg_nms.engine（已移入 trt/）。模板同 MNN，`model_path("yolo11n.engine", "trt")`，`baseline_dir("trt")`。

TRT 需 GPU RuntimeOption：
```cpp
static RuntimeOption trt_cpu_option() {
    RuntimeOption opt;
    opt.use_gpu(0);
    opt.use_trt_backend();
    return opt;
}
```
TRT 测试用 `trt_cpu_option()`（实际是 GPU）。

- [ ] **Step 4: 编译 + 验证**

```powershell
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && cd /d E:\CLionProjects\ModelDeploy && cmake --build build --target test_modeldeploy"
cd build\bin
$env:PATH = "E:\CLionProjects\ModelDeploy\build\bin;" + $env:PATH
$env:TEST_DATA_DIR = "E:\CLionProjects\ModelDeploy"
.\test_modeldeploy.exe "[backend:mnn]" 2>&1 | Select-Object -Last 4
.\test_modeldeploy.exe "[backend:trt]" 2>&1 | Select-Object -Last 4
```
注意：mnn/trt 基线尚未生成，self 对比 skip，但 ORT 基准对比会跑（可能因后端差异 FAIL——这是**预期的**，Task 5 生成基线后调整。或者先只在 baseline 存在时跑基准对比）。

**关键决策**：为避免 ORT 基准对比在基线未生成时误报，基准对比也应 `if (fs::exists(ort_file))` 才跑。这样 Task 4 完成后 mnn/trt 测试因无基线全 skip，验证编译通过即可。

- [ ] **Step 5: 提交**

```bash
git add tests/baseline_compare.cpp
git commit -m "test: 新增 MNN/TRT 跨后端回归 TEST_CASE（自对比 + ORT 基准对比）"
```

---

### Task 5: 生成 mnn/trt 后端基线 + 全量验证

**Files:**
- Create: `tests/baselines/mnn/*.json`、`tests/baselines/trt/*.json`

**Interfaces:**
- Consumes: Task 3 的 `baseline_collect --backend`
- Produces: mnn/trt 后端基线文件

- [ ] **Step 1: 用 collect 生成 mnn 基线**

```powershell
cd E:\CLionProjects\ModelDeploy\build\bin
$env:PATH = "E:\CLionProjects\ModelDeploy\build\bin;" + $env:PATH
$td = "E:\CLionProjects\ModelDeploy\test_data"
$out = "E:\CLionProjects\ModelDeploy\tests\baselines"
.\baseline_collect.exe --model $td\test_models\mnn\yolo11n.mnn --image $td\test_images\test_detection0.jpg --out $out\mnn --backend mnn --type det
.\baseline_collect.exe --model $td\test_models\mnn\yolo11n_nms.mnn --image $td\test_images\test_detection0.jpg --out $out\mnn --backend mnn --type det
```

- [ ] **Step 2: 用 collect 生成 trt 基线**

```powershell
.\baseline_collect.exe --model $td\test_models\trt\yolo11n.engine --image $td\test_images\test_detection0.jpg --out $out\trt --backend trt --type det
.\baseline_collect.exe --model $td\test_models\trt\yolo11n_nms.engine --image $td\test_images\test_detection0.jpg --out $out\trt --backend trt --type det
.\baseline_collect.exe --model $td\test_models\trt\yolo11n-seg_nms.engine --image $td\test_images\test_person.jpg --out $out\trt --backend trt --type seg
```

- [ ] **Step 3: 跑回归测试验证**

```powershell
cd E:\CLionProjects\ModelDeploy\build\bin
$env:TEST_DATA_DIR = "E:\CLionProjects\ModelDeploy"
.\test_modeldeploy.exe "[regression]" 2>&1 | Select-Object -Last 5
```
Expected: 全过。若有 FAIL，分析是后端真实差异（记录到测试报告）还是 bug。

- [ ] **Step 4: 提交**

```bash
git add tests/baselines/mnn tests/baselines/trt
git commit -m "test: 生成 MNN/TRT 后端回归基线"
```

---

### Task 6: yolo 系多后端模型转换

**Files:**
- Create: `test_data/test_models/trt/*.engine`、`test_data/test_models/mnn/*.mnn`（转换产物）

**Interfaces:**
- Consumes: Task 1 的 onnx/ 目录
- Produces: yolo 系全模型的多后端格式

- [ ] **Step 1: TRT 转换（本机 trtexec，RTX 4060 Ti）**

```powershell
$trtexec = "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.9.0.34\bin\trtexec.exe"
$onnx = "E:\CLionProjects\ModelDeploy\test_data\test_models\onnx"
$trt = "E:\CLionProjects\ModelDeploy\test_data\test_models\trt"
& $trtexec --onnx=$onnx\yolo11n-cls.onnx --saveEngine=$trt\yolo11n-cls.engine --fp16
& $trtexec --onnx=$onnx\yolo11n-obb.onnx --saveEngine=$trt\yolo11n-obb.engine --fp16
& $trtexec --onnx=$onnx\yolo11n-obb_nms.onnx --saveEngine=$trt\yolo11n-obb_nms.engine --fp16
& $trtexec --onnx=$onnx\yolo11n-pose.onnx --saveEngine=$trt\yolo11n-pose.engine --fp16
& $trtexec --onnx=$onnx\yolo11n-pose_nms.onnx --saveEngine=$trt\yolo11n-pose_nms.engine --fp16
& $trtexec --onnx=$onnx\yolo11n-seg.onnx --saveEngine=$trt\yolo11n-seg.engine --fp16
```
Expected: 6 个新 engine 生成。已有 yolo11n/yolo11n_nms/yolo11n-seg_nms engine 已移入 trt/。

- [ ] **Step 2: MNN 转换**

MNNConvert 需先安装：
```powershell
pip install MNN 2>&1 | Select-Object -Last 2
# 或从 MNN 官方 GitHub 下载 MNNConvert
```
若本机无法安装，改用服务器或记录为 skip（不阻塞）。

转换命令（MNNConvert 标准用法）：
```bash
MNNConvert -f ONNX --modelFile yolo11n.onnx --MNNModel yolo11n.mnn --bizCode biz
```
为 5 个 yolo 模型（det/cls/obb/pose/seg 各含 nms 版）转换。

- [ ] **Step 3: bmodel 转换（Sophgo 服务器）**

服务器 172.168.100.70（linaro/linaro），用 `tools/docker/sophgo/convert.sh`。需先将 onnx 上传服务器，转换后拉回。
```bash
# 在服务器 docker 内：
docker run --rm -it -v <onnx目录>:/conv tpuc_dev:1.27 bash /conv/convert.sh \
    --onnx yolo11n.onnx --name yolo11n --shapes "[[1,3,640,640]]" --chip bm1688 --quantize F16 --out yolo11n_bm1688.bmodel
```

- [ ] **Step 4: 提交转换产物**

```bash
git add test_data/test_models/trt/*.engine test_data/test_models/mnn/*.mnn test_data/test_models/sophgo/*.bmodel
git commit -m "test_data: yolo 系多后端模型转换（trt/mnn/bmodel）"
```
注意：大二进制文件可能触发 git 大小限制，若超限则用 git-lfs 或跳过提交（记录路径）。

---

### Task 7: sophgo 基线 + 服务器验证（可选，需 TPU）

**Files:**
- Create: `tests/baselines/sophgo/*.json`

**Interfaces:**
- Consumes: Task 6 的 bmodel + 服务器 Sophgo runtime
- Produces: sophgo 后端基线

- [ ] **Step 1: 上传 bmodel + 基线收集器到服务器**

```powershell
# 用 paramiko sftp（C:\Users\aichao\AppData\Local\Temp\opencode\sftp_put.py）
# 上传 bmodel 和 baseline_collect.exe（Linux 版需服务器重新编译）
```

- [ ] **Step 2: 服务器生成 sophgo 基线**

```bash
./baseline_collect --model bmodel/yolo11n.bmodel --image test_obb1.jpg --out baselines/sophgo --backend sophgo --type det
```

- [ ] **Step 3: 拉回基线 + 提交**

```bash
git add tests/baselines/sophgo
git commit -m "test: 生成 Sophgo 后端回归基线"
```

- [ ] **Step 4: 全量回归确认**

```powershell
cd E:\CLionProjects\ModelDeploy\build
ctest --output-on-failure 2>&1 | Select-Object -Last 5
```
Expected: 全部通过（含 [regression] 及 mnn/trt 后端）

---

## Self-Review

**Spec coverage:**
- 全覆盖迁移（Task 1）✓
- baseline_compare backend 路由（Task 2）✓
- baseline_collect --backend（Task 3）✓
- 跨后端 TEST_CASE mnn/trt（Task 4）✓
- 生成 mnn/trt 基线（Task 5）✓
- yolo 系转换（Task 6）✓
- sophgo 基线（Task 7）✓

**待确认点（实现时处理，不阻塞计划）：**
- Task 6 MNNConvert 本机可用性——不可用则服务器转换或 skip
- Task 6/7 bmodel 需 Sophgo TPU 服务器，本机无法执行
- TRT 测试的 RuntimeOption：use_trt_backend + use_gpu（TRT 必须 GPU）
- 大二进制（engine/bmodel）提交：若超 git 限制用 git-lfs 或 .gitignore 记录路径
- 迁移后 face/、ocr/ 原目录可能留非 onnx 文件（.txt 等），确认哪些保留
