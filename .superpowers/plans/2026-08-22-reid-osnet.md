# 行人 Re-ID（OSNet，512-d Embedding）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 ModelDeploy 增加行人重识别能力——OSNet 从裁剪行人图提取 512 维 L2 归一化特征，配合内存 `ReIdGallery` 做 top-k 余弦匹配，可与追踪器 `TrackResult.feature` 搭配。

**Architecture:** 新 `csrc/vision/reid/` 模块：独立 `ReID` 模型类（镜像 `SeetaFaceID`），专用 preprocessor（256×128 CHW）+ L2 后处理；内存 `ReIdGallery`（enroll/match top-k）。按现有模型模板贯穿六面。

**Tech Stack:** C++17、pybind11、OnnxRuntime/MNN/TRT/Sophgo（BaseBackend）、Catch2。

**Spec:** `docs/superpowers/specs/2026-08-22-reid-osnet-design.md`

## Global Constraints

- 独立 `ReID` 模型类（不复用分类包装）；`ReIdResult { std::vector<float> embedding; }`（512-d，已 L2 归一化）。
- 预处理器：固定 `256x128`，`BGR→RGB CHW`，ImageNet 归一化（alpha/beta），mirror `csrc/vision/tracking/reid_extractor.cpp:50-76`。
- 后处理器：512-d 输出 → `embedding` + `utils::l2_normalize`（`utils.cpp:471`）。
- Gallery：内存 `map<string, vector<float>>`，`enroll`（覆盖同 label）、`match(embedding,k)` 用 `utils::compute_similarity`（`cosine_dist=1-dot`）降序 top-k。
- 权重外链：modelscope `test_data/test_models/onnx/osnet_x1_0.onnx`，不在仓库。
- 测试数据经 `TEST_DATA_DIR` 定位。
- 命名枚举：C++ `ReID`、CAPI `MD_MODEL_REID`、C# `ReIdModel`、Rust `ReID`/`ModelKind::ReId`、Python `ReID`。
- `MDModelKind` 新值追加在 `MD_MODEL_COUNT` 之前（ABI 兼容）。
- Sophgo int8 bmodel 通常 batch=1 静态形状，按需 `set_cls_batch_size(1)` 类调整。
- 不做 CAPI/C#/Rust 追踪器 per-detection feature 注入（明确不做）。
- Windows：.bat 包裹 vcvars64 经 `cmd /c` 构建；`bash.exe` 是 WSL bash。

---

### Task 1: C++ 核心 — 结果类型 + ReID 预/后处理器 + 模型类

**Files:**
- Create: `csrc/vision/reid/reid.h`
- Create: `csrc/vision/reid/reid.cpp`
- Create: `csrc/vision/reid/preprocessor.h`
- Create: `csrc/vision/reid/preprocessor.cpp`
- Create: `csrc/vision/reid/postprocessor.h`
- Create: `csrc/vision/reid/postprocessor.cpp`
- Modify: `csrc/vision/common/result.h`（加 `ReIdResult`）
- Modify: `csrc/vision.h`（include reid.h）

**Interfaces:**
- Consumes: `BaseModel`（`csrc/base_model.h`）、`RuntimeOption`、`utils::l2_normalize`（`csrc/vision/utils.h`）、`Tensor`。
- Produces: `vision::reid::ReID`（`predict`/`batch_predict`/`get_preprocessor`/`get_postprocessor`/`clone`）、`ReIDPreprocessor`、`ReIDPostprocessor`、`ReIdResult`（`result.h`）。

- [ ] **Step 1: `csrc/vision/common/result.h` 加 ReIdResult（人脸结果旁）**

```cpp
struct ReIdResult {
    std::vector<float> embedding;  //!< 已 L2 归一化的 512-d 特征
};
```

（加在 `FaceRecognitionResult` 附近，`#include <vector>` 已具备。）

- [ ] **Step 2: 创建 `csrc/vision/reid/preprocessor.h`**

```cpp
#pragma once
#include <vector>
#include "vision/common/image_data.h"
#include "vision/runtime/tensor.h"

namespace modeldeploy::vision::reid {
    class MODELDEPLOY_CXX_EXPORT ReIDPreprocessor {
    public:
        ReIDPreprocessor();
        bool run(const std::vector<ImageData>& images,
                 std::vector<Tensor>* output_tensors);
        void set_size(int w, int h) { width_ = w; height_ = h; }
        std::pair<int,int> size() const { return {width_, height_}; }
    private:
        int width_ = 128;   // 内部按 CHW; OSNet 输入 256x128 (HxW)
        int height_ = 256;
    };
}
```

- [ ] **Step 3: 创建 `csrc/vision/reid/preprocessor.cpp`**

镜像 `csrc/vision/classification/preprocessor.cpp` 的 ImageNet 归一化与 CHW，但尺寸 256×128、BGR→RGB：

```cpp
#include "vision/reid/preprocessor.h"
#include <opencv2/imgproc.hpp>
#include "vision/utils.h"

namespace modeldeploy::vision::reid {
    ReIDPreprocessor::ReIDPreprocessor() = default;

    bool ReIDPreprocessor::run(const std::vector<ImageData>& images,
                               std::vector<Tensor>* output_tensors) {
        // 参考 classification/preprocessor.cpp 的实现：
        //  1) 各图 resize 到 (height_, width_) = 256x128
        //  2) BGR->RGB, 转 float, 归一化 (alpha=1/(255*sigma), beta=-mu/sigma, ImageNet)
        //  3) 拼成 NCHW 连续 Tensor -> output_tensors
        // 简略示意（实际按 classification 实现填充）：
        (void)images; (void)output_tensors;
        return true;
    }
}
```

> **重要**：Step 3 不得留空——实际代码需逐行镜像 `csrc/vision/classification/preprocessor.cpp:42-98`（含 alpha/beta 表 224→256×128、CHW 布局、GPU/CUDA 后端选择留待 `initialize`）。实现者以该文件为蓝本完整写出。

- [ ] **Step 4: 创建 `csrc/vision/reid/postprocessor.h`**

```cpp
#pragma once
#include <vector>
#include "vision/runtime/tensor.h"
#include "vision/common/result.h"

namespace modeldeploy::vision::reid {
    class MODELDEPLOY_CXX_EXPORT ReIDPostprocessor {
    public:
        bool run(const std::vector<Tensor>& inputs,
                 std::vector<std::vector<ReIdResult>>* results);
    };
}
```

- [ ] **Step 5: 创建 `csrc/vision/reid/postprocessor.cpp`**

镜像 `csrc/vision/face/face_rec/postprocessor.cpp:12-33`（reshape + `utils::l2_normalize`）：

```cpp
#include "vision/reid/postprocessor.h"
#include "vision/utils.h"

namespace modeldeploy::vision::reid {
    bool ReIDPostprocessor::run(const std::vector<Tensor>& inputs,
                                std::vector<std::vector<ReIdResult>>* results) {
        const auto& in = inputs[0];
        // in.shape: {B, C} 或 {B,1,1,C}; flatten 到 {B, N}
        // 对每个 item: ReIdResult{ embedding }; utils::l2_normalize(embedding);
        // results->at(i) = { item }; return true;
        (void)in; (void)results;
        return true;
    }
}
```

> **重要**：Step 5 实际实现需从 `inputs[0]` 提取数值（`Tensor` 数据 API 依 `reid_extractor.cpp`），逐样本拷贝 `embedding` 并 `utils::l2_normalize`。以 `face_rec/postprocessor.cpp` 为实现蓝本完整写出（不得留空）。

- [ ] **Step 6: 创建 `csrc/vision/reid/reid.h`**

```cpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "base_model.h"
#include "runtime_option.h"
#include "vision/reid/preprocessor.h"
#include "vision/reid/postprocessor.h"
#include "vision/common/result.h"

namespace modeldeploy::vision::reid {
    class MODELDEPLOY_CXX_EXPORT ReID : public BaseModel {
    public:
        explicit ReID(const std::string& model_file,
                      const RuntimeOption& option = RuntimeOption());

        bool predict(const ImageData& img,
                     std::vector<ReIdResult>* results,
                     TimerArray* timer = nullptr) const;

        bool batch_predict(const std::vector<ImageData>& imgs,
                           std::vector<std::vector<ReIdResult>>* results,
                           TimerArray* timer = nullptr) const;

        ReIDPreprocessor* get_preprocessor() { return &preprocessor_; }
        const ReIDPreprocessor* get_preprocessor() const { return &preprocessor_; }
        ReIDPostprocessor* get_postprocessor() { return &postprocessor_; }
        const ReIDPostprocessor* get_postprocessor() const { return &postprocessor_; }

        std::unique_ptr<ReID> clone() const;

    private:
        bool initialize();
        void setup_processor_backend();

        ReIDPreprocessor preprocessor_;
        ReIDPostprocessor postprocessor_;
        mutable std::vector<Tensor> reused_input_tensors_;
        mutable std::vector<Tensor> reused_output_tensors_;
    };
}
```

> 字段命名与 `Classification`/`SeetaFaceID` 对齐（`reused_input_tensors_` 等）。

- [ ] **Step 7: 创建 `csrc/vision/reid/reid.cpp`**

镜像 `csrc/vision/classification/classification.cpp` 与 `seetaface.cpp`：

```cpp
#include "vision/reid/reid.h"
#include "vision/utils.h"

namespace modeldeploy::vision::reid {
    ReID::ReID(const std::string& model_file, const RuntimeOption& option)
        : BaseModel(model_file, option) {
        initialize();
    }

    bool ReID::initialize() {
        if (!init_runtime()) return false;
        setup_processor_backend();
        return true;
    }

    void ReID::setup_processor_backend() {
        // 参考 Classification::initialize / SeetaFaceID: 按 device/backend 设置预处理器内核(CPU/CUDA/Sophgo)
    }

    bool ReID::predict(const ImageData& img,
                       std::vector<ReIdResult>* results,
                       TimerArray* timer) const {
        std::vector<ImageData> batch{img};
        std::vector<std::vector<ReIdResult>> tmp;
        if (!batch_predict(batch, &tmp, timer)) return false;
        *results = std::move(tmp[0]);
        return true;
    }

    bool ReID::batch_predict(const std::vector<ImageData>& imgs,
                             std::vector<std::vector<ReIdResult>>* results,
                             TimerArray* timer) const {
        results->clear();
        results->resize(imgs.size());
        if (!preprocessor_.run(imgs, &reused_input_tensors_)) return false;
        if (!infer(reused_input_tensors_, &reused_output_tensors_, timer)) return false;
        return postprocessor_.run(reused_output_tensors_, results);
    }

    std::unique_ptr<ReID> ReID::clone() const { return std::make_unique<ReID>(*this); }
}
```

> 确认 `infer`/`init_runtime`/`reused_*` 为 `BaseModel` 成员（`csrc/base_model.h`），与 `Classification` 用法一致。

- [ ] **Step 8: `csrc/vision.h` include 区加**

```cpp
#include "vision/reid/reid.h"
```

- [ ] **Step 9: 验证编译**

`.bat` 构建 `build_tdc_gpu`。Expected: 构建成功（自动 GLOB）。

- [ ] **Step 10: Commit**

```bash
git add csrc/vision/reid csrc/vision/common/result.h csrc/vision.h
git commit -m "feat(reid): ReID model core + pre/postprocessor + result"
```

---

### Task 2: C++ 测试 + ReIdGallery

**Files:**
- Create: `csrc/vision/reid/gallery.h`
- Create: `csrc/vision/reid/gallery.cpp`
- Create: `tests/test_reid.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `ReID::predict`（Task 1）、`utils::compute_similarity`/`l2_normalize`。
- Produces: `reid::ReIdGallery`（`enroll`/`match`/`remove`/`clear`/`size`）。

- [ ] **Step 1: 创建 `csrc/vision/reid/gallery.h`**

```cpp
#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace modeldeploy::vision::reid {
    /*! 内存式 Re-ID 特征库：注册 label->embedding，top-k 余弦匹配 */
    class MODELDEPLOY_CXX_EXPORT ReIdGallery {
    public:
        void clear() { gallery_.clear(); }
        void enroll(const std::string& label, const std::vector<float>& embedding);  // 覆盖同 label
        std::vector<bool> remove(const std::string& label);
        std::vector<std::pair<std::string, float>> match(const std::vector<float>& embedding, int k) const;
        size_t size() const { return gallery_.size(); }

    private:
        std::map<std::string, std::vector<float>> gallery_;
    };
}
```

- [ ] **Step 2: 创建 `csrc/vision/reid/gallery.cpp`**

```cpp
#include "vision/reid/gallery.h"
#include "vision/utils.h"
#include <algorithm>

namespace modeldeploy::vision::reid {
    void ReIdGallery::enroll(const std::string& label, const std::vector<float>& embedding) {
        gallery_[label] = embedding;   // 覆盖同 label
    }

    std::vector<bool> ReIdGallery::remove(const std::string& label) {
        return { gallery_.erase(label) > 0 };
    }

    std::vector<std::pair<std::string,float>> ReIdGallery::match(
            const std::vector<float>& embedding, int k) const {
        std::vector<std::pair<float,std::string>> scored;
        scored.reserve(gallery_.size());
        for (const auto& [label, feat] : gallery_) {
            scored.emplace_back(utils::compute_similarity(feat, embedding), label);
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        if (k > 0 && static_cast<size_t>(k) < scored.size())
            scored.resize(k);
        std::vector<std::pair<std::string,float>> out;
        out.reserve(scored.size());
        for (auto& [s, label] : scored)
            out.emplace_back(std::move(label), s);
        return out;
    }
}
```

- [ ] **Step 3: 创建 `tests/test_reid.cpp`（`[reid]`）**

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <string>
#include "vision/reid/reid.h"
#include "vision/reid/gallery.h"

using namespace modeldeploy::vision;

namespace {
    std::string test_dir() {
        const char* v = std::getenv("TEST_DATA_DIR");
        return v ? std::string(v) : std::string("./");
    }
}

TEST_CASE("ReIdGallery top-k cosine match", "[reid]") {
    reid::ReIdGallery g;
    std::vector<float> a(512, 0.0f), b(512, 0.0f), q(512, 0.0f);
    a[0] = 1.0f; b[1] = 1.0f; q[0] = 0.9f; q[1] = 0.1f;  // q 更接近 a
    utils::l2_normalize(a); utils::l2_normalize(b); utils::l2_normalize(q);
    g.enroll("A", a);
    g.enroll("B", b);
    REQUIRE(g.size() == 2);
    auto top = g.match(q, 1);
    REQUIRE(top.size() == 1);
    REQUIRE(top[0].first == "A");
}

TEST_CASE("ReID predict produces 512-d L2 embedding", "[reid]") {
    RuntimeOption opt; opt.use_ort_backend();
    const auto path = test_dir() + "/test_data/test_models/onnx/osnet_x1_0.onnx";
    reid::ReID model(path, opt);
    // 构造 256x128 图：ImageData; predict -> results
    // 断言 embedding.size()==512 且 norm≈1（权重缺失时 SKIP 占位）
}
```

> **权重可用性**：OSNet ONNX 外链。若本机无权重，`[reid]` 主路径用 SKIP 占位并注明 modelscope 下载；Gallery 单测不依赖权重，始终可跑。

- [ ] **Step 4: `tests/CMakeLists.txt`** TEST_SOURCES 加 `test_reid.cpp`。

- [ ] **Step 5: 构建 + 运行**

`test_modeldeploy.exe "[reid]"`。Expected: Gallery 单测通过；predict 测试（有权重时 512-d/L2 通过，缺权重时 SKIP）。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/reid/gallery.h csrc/vision/reid/gallery.cpp tests/test_reid.cpp tests/CMakeLists.txt
git commit -m "feat(reid): ReIdGallery + reid tests"
```

---

### Task 3: pybind 绑定

**Files:**
- Create: `csrc/pybind/vision/reid_pybind.cpp`
- Modify: `csrc/pybind/vision/vision_pybind.cpp`

**Interfaces:**
- Consumes: `reid::ReID`、`reid::ReIdGallery`（Task 1/2）。
- Produces: Python `vision.ReID`、`ReIdResult`、`ReIdGallery`。

- [ ] **Step 1: 创建 `csrc/pybind/vision/reid_pybind.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind/utils/utils.h"
#include "vision/reid/reid.h"
#include "vision/reid/gallery.h"

namespace modeldeploy::vision {
    void bind_reid(const pybind11::module& m) {
        pybind11::class_<reid::ReIdResult>(m, "ReIdResult")
            .def(pybind11::init<>())
            .def_readwrite("embedding", &reid::ReIdResult::embedding);

        pybind11::class_<reid::ReID>(m, "ReID")
            .def(pybind11::init<const std::string&, const RuntimeOption&>(),
                 pybind11::arg("model_file"), pybind11::arg("option"))
            .def("predict",
                 [](const reid::ReID& self, const pybind11::array& im) {
                     auto cv = pyarray_to_cv_mat(im);
                     std::vector<reid::ReIdResult> out;
                     ImageData img(cv);
                     if (!self.predict(img, &out))
                         throw std::runtime_error("ReID predict failed");
                     return out;
                 }, pybind11::arg("image"));

        pybind11::class_<reid::ReIdGallery>(m, "ReIdGallery")
            .def(pybind11::init<>())
            .def("enroll", &reid::ReIdGallery::enroll,
                 pybind11::arg("label"), pybind11::arg("embedding"))
            .def("match", &reid::ReIdGallery::match,
                 pybind11::arg("embedding"), pybind11::arg("k"))
            .def("remove", &reid::ReIdGallery::remove, pybind11::arg("label"))
            .def("clear", &reid::ReIdGallery::clear)
            .def("size", &reid::ReIdGallery::size);
    }
}
```

- [ ] **Step 2: `vision_pybind.cpp`** 声明 + 调用 `bind_reid`（同 hand）。

- [ ] **Step 3: 构建 + 冒烟**

`.bat` 构建 `build_py`；`python -c "from modeldeploy.vision import ReID, ReIdGallery; print(ReID, ReIdGallery)"`。
Expected: 打印两个类。

- [ ] **Step 4: Commit**

```bash
git add csrc/pybind/vision/reid_pybind.cpp csrc/pybind/vision/vision_pybind.cpp
git commit -m "feat(reid): pybind ReID + ReIdGallery"
```

---

### Task 4: CAPI — MD_MODEL_REID

**Files:**
- Modify: `capi/md_capi.h`、`capi/md_capi.cpp`
- Modify: `tests/test_capi.cpp`（契约）

**Interfaces:**
- Consumes: `reid::ReID`（Task 1）。
- Produces: `MD_MODEL_REID`、`md_result_reid_embedding`。

- [ ] **Step 1: `md_capi.h`** `MD_MODEL_COUNT` 前加 `MD_MODEL_REID`。

- [ ] **Step 2: `md_capi.h`** 加 `md_result_reid_embedding`（镜像 `md_result_face_embedding` :380-388）：

```c
MDStatus md_result_reid_embedding(MDModelHandle model, const float* embedding, size_t* len);
```

- [ ] **Step 3: `md_capi.cpp`** 分支：`md_model_create` 加 `case MD_MODEL_REID` 构造 `new vision::reid::ReID(...)`；`destroy`/`clone` 对应；`md_model_predict` 推断后按 face-rec 模式（输出 `ReIdResult.embedding` 存于结果）；实现 `md_result_reid_embedding`（镜像 `md_result_face_embedding` :2421）。

> 逐行对齐现有 `MD_MODEL_FACE_RECOGNITION`（SeetaFace）的完整注册与结果提取，研究已列：create :771、params、predict :1562-1571、getter :2421-2474。

- [ ] **Step 4: `test_capi.cpp`** 加 `[reid]` 契约：create `MD_MODEL_REID`、predict、`md_result_reid_embedding`（校验 len；缺权重 SKIP）。

- [ ] **Step 5: 构建 + `[capi]`**

`test_modeldeploy.exe "[capi]"`。Expected: 通过，无回归。

- [ ] **Step 6: Commit**

```bash
git add capi/md_capi.h capi/md_capi.cpp tests/test_capi.cpp
git commit -m "feat(reid): CAPI MD_MODEL_REID + embedding getter"
```

---

### Task 5: C# 绑定 — ReIdModel

**Files:**
- Modify: `csharp/ModelDeploy/Models.cs`（加 `ReIdModel`，镜像 `FaceRecModel`）
- Modify: `csharp/ModelDeploy/enum_varaibles.cs`（加 `MD_MODEL_REID`）
- Modify: `csharp/ModelDeploy/NativeMethods.cs`（`md_result_reid_embedding`）
- Modify: `csharp/ModelDeployUnitTest/AllModelsTests.cs`（`ReId_Works`）

- [ ] **Step 1:** `enum_varaibles.cs` 加 `MD_MODEL_REID`。
- [ ] **Step 2:** `Models.cs` 加 `ReIdModel`（`Predict` 返回含 `Embedding`（float[]）的结果；镜像 `FaceRecModel` :426-444）。
- [ ] **Step 3:** `NativeMethods.cs` 加 `md_result_reid_embedding` DllImport。
- [ ] **Step 4:** `AllModelsTests.cs` 加 `ReId_Works`。
- [ ] **Step 5:** `dotnet build` + `dotnet test`（4 个 audio 失败为已知预存，不归因）。

- [ ] **Step 6: Commit**

```bash
git add csharp/ModelDeploy csharp/ModelDeployUnitTest/AllModelsTests.cs
git commit -m "feat(reid): C# ReIdModel binding + tests"
```

---

### Task 6: Rust 绑定 — ReID

**Files:**
- Modify: `rust/modeldeploy/src/ffi.rs`（`MD_MODEL_REID` + `md_result_reid_embedding`）
- Modify: `rust/modeldeploy/src/model.rs`（`ResultType for ReID` + `model_wrapper!` + `reid()`）
- Modify: `rust/modeldeploy/src/types.rs`（`ModelKind::ReId` + `ReIdResult`）
- Modify: `rust/modeldeploy/src/lib.rs`
- Modify: `rust/modeldeploy/tests/integration_test.rs`（`test_reid`）

- [ ] **Step 1:** `types.rs` 加 `ModelKind::ReId` + `ReIdResult { embedding: Vec<f32> }`。
- [ ] **Step 2:** `ffi.rs` 加 `MD_MODEL_REID` + `md_result_reid_embedding` extern。
- [ ] **Step 3:** `model.rs` 镜像 face embedding（:475-519）封装 `reid()`。
- [ ] **Step 4:** `cargo build` + `cargo clippy -- -D warnings`（clean）+ `cargo test`（`test_reid` 通过，无回归）。

- [ ] **Step 5: Commit**

```bash
git add rust/modeldeploy
git commit -m "feat(reid): Rust ReID binding + test"
```

---

### Task 7: Demo + Docs

**Files:**
- Create: `examples/demo_reid/`（`demo_reid.cpp` + `CMakeLists.txt`）
- Modify: `examples/CMakeLists.txt`、`examples/EXAMPLES.md`、`README.md`

- [ ] **Step 1: `examples/demo_reid/demo_reid.cpp`**：加载 OSNet，`predict` 两批裁剪图得到 embedding → `ReIdGallery.enroll("A",e1)`/`enroll("B",e2)` → `match(queryEmbedding,1)` 打印 `(label,score)`。
- [ ] **Step 2: `CMakeLists.txt`**：单 `add_executable` + 链接 `LIBRARY_NAME` + `OpenCV_LIBS`。
- [ ] **Step 3:** `examples/CMakeLists.txt` 加 `add_subdirectory(demo_reid)`。
- [ ] **Step 4:** 构建 + 运行（无权重时至少构建通过 + Usage 路径）。
- [ ] **Step 5:** `EXAMPLES.md`/`README.md` 加"行人 Re-ID"条目。
- [ ] **Step 6: Commit**

```bash
git add examples/demo_reid examples/CMakeLists.txt examples/EXAMPLES.md README.md
git commit -m "feat(reid): demo_reid + docs"
```

---

### Task 8: 全量构建 + 回归验证

- [ ] **Step 1:** `test_modeldeploy.exe "[reid]"`、`"[capi]"`、`"[core]"`、`"[tracking]"`、`"[barcode]"` 全 PASS，tracking/barcode 无回归。
- [ ] **Step 2:** 跨后端语义：`ReID` 用 BaseModel/BaseBackend，天然支持 ORT/MNN/TRT/Sophgo（记录）。
- [ ] **Step 3:** Python import + `ReID`/`ReIdGallery` 冒烟；C# `ReId_Works`；Rust `test_reid`；`demo_reid` 运行。
- [ ] **Step 4:** 收尾 commit（仅当有变更）。

---

## Self-Review 核查

- **Spec 覆盖**：独立 ReID 模型 + 256×128/CHW/L2 → Task 1；ReIdGallery → Task 2；六面集成 → Task 3/4/5/6/7；测试/回归 → Task 2/8；明确不做（追踪器 feature 注入/持久化 Gallery）→ 设计未含，符合。
- **Placeholder**：Step 3/5 的 pre/postprocessor 明确指向实现在蓝本（classification/face_rec）完整写出，不得留空；权重缺失测试用 SKIP 占位并注明。
- **类型一致性**：`ReID`/`ReIdResult.embedding`（`vector<float>`/`Vec<f32>`/`float[]`/`list[float]`）、`MD_MODEL_REID`/`MDModelKind.MD_MODEL_REID`/`ModelKind::ReId`、`enroll`/`match` 各 Task 一致。
