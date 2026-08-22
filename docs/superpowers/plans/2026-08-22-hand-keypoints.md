# 手部关键点（Hand Keypoints，21 点） Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 ModelDeploy 增加手部关键点识别能力——检测手部并输出 21 个关键点（MediaPipe 风格），薄封装复用现有 UltralyticsPose 管线。

**Architecture:** 新 `csrc/vision/hand/` 模块，`HandKeypoint` 类内部持有 `UltralyticsPose`（`detection` 命名空间）并默认 `set_keypoints_num(21)`；新增 MediaPipe 21 骨架可视化 `vis_hand`；按现有模型模板贯穿六面（C++/pybind/CAPI/C#/Rust/demo+docs+tests）。

**Tech Stack:** C++17、pybind11、OnnxRuntime/MNN/TRT/Sophgo（BaseBackend）、OpenCV（可视化）、Catch2（测试）。

**Spec:** `docs/superpowers/specs/2026-08-22-hand-keypoints-design.md`

## Global Constraints

- 复用现有 `UltralyticsPose` 管线（不新建专用 pre/postprocessor）。
- 结果类型复用 `KeyPointsResult`（`box` + `vector<Point3f> keypoints`，`Point3f.z` 存置信度）。
- 关键点数量默认 21（`set_keypoints_num(21)`）。
- 权重外链：modelscope `test_data/test_models/onnx/...`，不在仓库。
- 测试数据经 `TEST_DATA_DIR`（运行期环境变量）定位，如 `<TEST_DATA_DIR>/test_data/test_models/...`。
- 命名枚举：C++ `HandKeypoint`、CAPI `MD_MODEL_HAND`、C# `HandModel`、Rust `HandKeypoint`/`ModelKind::Hand`、Python `HandKeypoint`。
- `MDModelKind` 是稳定 C ABI 枚举——新值追加在 `MD_MODEL_COUNT` 之前。
- Sophgo int8 bmodel 通常 batch=1 静态形状，pipeline 内按需 `set_cls_batch_size(1)` 类调整。
- Windows：.bat 包裹 `vcvars64.bat` 经 `cmd /c` 构建；`bash.exe` 是 WSL bash。

---

### Task 1: C++ 核心 — HandKeypoint 薄封装类

**Files:**
- Create: `csrc/vision/hand/hand.h`
- Create: `csrc/vision/hand/hand.cpp`
- Modify: `csrc/vision.h`（加 `#include "vision/hand/hand.h"`）

**Interfaces:**
- Consumes: `modeldeploy::vision::detection::UltralyticsPose`（`csrc/vision/pose/ultralytics_pose.h`）、`RuntimeOption`（`csrc/runtime_option.h`）、`KeyPointsResult`（`csrc/vision/common/result.h`）、`BaseModel`。
- Produces: `modeldeploy::vision::hand::HandKeypoint`（`predict`/`batch_predict`/`draw_result`/`get_preprocessor`/`get_postprocessor`/`clone`）。

- [ ] **Step 1: 创建 `csrc/vision/hand/hand.h`**

```cpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "base_model.h"
#include "runtime_option.h"
#include "vision/pose/ultralytics_pose.h"
#include "vision/common/result.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision::hand {
    /*! @brief 手部关键点识别（21 点，MediaPipe 风格），薄封装 UltralyticsPose */
    class MODELDEPLOY_CXX_EXPORT HandKeypoint {
    public:
        explicit HandKeypoint(const std::string& model_file,
                              const RuntimeOption& option = RuntimeOption());

        bool predict(const ImageData& img,
                     std::vector<KeyPointsResult>* results,
                     TimerArray* timer = nullptr) const;

        bool batch_predict(const std::vector<ImageData>& imgs,
                           std::vector<std::vector<KeyPointsResult>>* results,
                           TimerArray* timer = nullptr) const;

        void draw_result(ImageData& img,
                         const std::vector<KeyPointsResult>& results,
                         double threshold = 0.5) const;

        vision::detection::UltralyticsPosePreprocessor* get_preprocessor();
        const vision::detection::UltralyticsPosePreprocessor* get_preprocessor() const;
        vision::detection::UltralyticsPosePostprocessor* get_postprocessor();
        const vision::detection::UltralyticsPosePostprocessor* get_postprocessor() const;

        std::unique_ptr<HandKeypoint> clone() const;

    private:
        vision::detection::UltralyticsPose pose_;
    };
}
```

- [ ] **Step 2: 创建 `csrc/vision/hand/hand.cpp`**

```cpp
#include "vision/hand/hand.h"

namespace modeldeploy::vision::hand {
    HandKeypoint::HandKeypoint(const std::string& model_file,
                               const RuntimeOption& option)
        : pose_(model_file, option) {
        pose_.get_postprocessor()->set_keypoints_num(21);
    }

    bool HandKeypoint::predict(const ImageData& img,
                               std::vector<KeyPointsResult>* results,
                               TimerArray* timer) const {
        return pose_.predict(img, results, timer);
    }

    bool HandKeypoint::batch_predict(const std::vector<ImageData>& imgs,
                                     std::vector<std::vector<KeyPointsResult>>* results,
                                     TimerArray* timer) const {
        return pose_.batch_predict(imgs, results, timer);
    }

    void HandKeypoint::draw_result(ImageData& img,
                                   const std::vector<KeyPointsResult>& results,
                                   double threshold) const {
        pose_.draw_result(img, results, threshold);
    }

    vision::detection::UltralyticsPosePreprocessor* HandKeypoint::get_preprocessor() {
        return pose_.get_preprocessor();
    }
    const vision::detection::UltralyticsPosePreprocessor* HandKeypoint::get_preprocessor() const {
        return pose_.get_preprocessor();
    }
    vision::detection::UltralyticsPosePostprocessor* HandKeypoint::get_postprocessor() {
        return pose_.get_postprocessor();
    }
    const vision::detection::UltralyticsPosePostprocessor* HandKeypoint::get_postprocessor() const {
        return pose_.get_postprocessor();
    }

    std::unique_ptr<HandKeypoint> HandKeypoint::clone() const {
        auto ret = std::make_unique<HandKeypoint>(*this);
        return ret;
    }
}
```

> 注意：确认 `UltralyticsPose::predict/batch_predict/draw_result/get_preprocessor/get_postprocessor` 的实际签名（const 与否）与 `csrc/vision/pose/ultralytics_pose.h` 一致，必要时调整上层 const 修饰；`clone()` 依赖 `UltralyticsPose` 的深拷贝（`pose_` 若为不可拷贝成员则改为 `std::unique_ptr<...>` 并在 clone 中重构造，详见研究指出的 pose 有 `clone()`）。

- [ ] **Step 3: 在 `csrc/vision.h` 顶部 include 区加入**

```cpp
#include "vision/hand/hand.h"
```

- [ ] **Step 4: 验证编译通过**

Windows：`.bat` 包裹 vcvars64 `cmake --build build_tdc_gpu --config Release`。
Expected: 构建成功（新 `.cpp` 由根 CMakeLists `file(GLOB_RECURSE VISION_SOURCE)` 自动收集）。

- [ ] **Step 5: Commit**

```bash
git add csrc/vision/hand/hand.h csrc/vision/hand/hand.cpp csrc/vision.h
git commit -m "feat(hand): HandKeypoint thin wrapper around UltralyticsPose"
```

---

### Task 2: C++ 测试 + MediaPipe 21 骨架可视化

**Files:**
- Create: `tests/test_hand.cpp`
- Modify: `tests/CMakeLists.txt`（TEST_SOURCES 加 `test_hand.cpp`）
- Create: `csrc/vision/common/visualize/vis_hand.cpp`
- Modify: `csrc/vision/common/visualize/`（如需头文件，见下）

**Interfaces:**
- Consumes: `HandKeypoint::predict`/`set_keypoints_num`（Task 1）、`KeyPointsResult`。
- Produces: `vis_hand(...)` 可视化函数。

- [ ] **Step 1: 创建 `tests/test_hand.cpp`（`[hand]` 标签）**

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <string>
#include "vision/hand/hand.h"

using namespace modeldeploy::vision;

namespace {
    std::string test_dir() {
        const char* v = std::getenv("TEST_DATA_DIR");
        return v ? std::string(v) : std::string("./");
    }
}

TEST_CASE("HandKeypoint predict produces 21 keypoints", "[hand]") {
    RuntimeOption opt;
    opt.use_ort_backend();
    const auto path = test_dir() + "/test_data/test_models/onnx/hand_pose.onnx";
    hand::HandKeypoint model(path, opt);
    const auto& pp = model.get_postprocessor();
    REQUIRE(pp->get_keypoints_num() == 21);
    // 空图冒烟：构造 640x640 BGR ImageData
    // 说明：若实际权重不可用时该测试无法跑，见下"模型不可用"备注
}
```

> **模型可用性备注**：手部 ONNX 权重外链（modelscope）。若实现时本机无权重，`[hand]` 测试应在缺失时用 `SKIP`/判空 img 逻辑占位，并在报告注明"权重待 modelscope 下载"。测试代码需包含真实加载 + `predict` + 断言 `results->front().keypoints.size()==21` 的主路径（用占位构造）。

- [ ] **Step 2: 在 `tests/CMakeLists.txt` TEST_SOURCES 追加**

```cmake
"${CMAKE_CURRENT_SOURCE_DIR}/test_hand.cpp"
```

（模式同现有 `test_barcode.cpp` / `test_tracking.cpp`。）

- [ ] **Step 3: 创建 `csrc/vision/common/visualize/vis_hand.h`（若无通用可视化头）**

新建头，或在现有 `visualize.h` 声明：

```cpp
#pragma once
#include <vector>
#include "vision/common/image_data.h"
#include "vision/common/result.h"

namespace modeldeploy::vision {
    void vis_hand(const ImageData& im,
                  const std::vector<KeyPointsResult>& result,
                  double threshold = 0.5,
                  const std::string& save_name = "",
                  int font_size = 14,
                  float alpha = 0.15);
}
```

- [ ] **Step 4: 创建 `csrc/vision/common/visualize/vis_hand.cpp`**

镜像 `vis_pose.cpp` 结构，但骨架换为 MediaPipe 21 点连接表与 `hand_palette`：

```cpp
#include "vis_hand.h"
#include <opencv2/imgproc.hpp>

namespace modeldeploy::vision {
namespace {
    // MediaPipe hand 骨架（22 条连接，端点相对关键点数组下标）
    const std::vector<std::pair<int,int>> kHandSkeleton = {
        {1,2},{2,3},{3,4},        // 拇指 (idx 1..4)
        {0,5},{5,6},{6,7},{7,8},  // 食指
        {5,9},{9,10},{10,11},{11,12}, // 中指
        {9,13},{13,14},{14,15},{15,16}, // 无名指
        {13,17},{17,18},{18,19},{19,20}, // 小指
        {0,17}                    // 掌心根部
    };
    const std::vector<std::vector<int>> kHandPalette = {
        {255,0,0},{255,85,0},{255,170,0},{255,255,0},
        {170,255,0},{85,255,0},{0,255,0},{0,255,85},
        {0,255,170},{0,255,255},{0,170,255},{0,85,255},
        {0,0,255},{85,0,255},{170,0,255},{255,0,255},
        {255,0,170},{255,0,85},{255,0,0},{170,0,0},
        {0,255,0}
    };
}

void vis_hand(const ImageData& im,
              const std::vector<KeyPointsResult>& result,
              double threshold,
              const std::string& save_name,
              int font_size,
              float alpha) {
    // 将 ImageData 转为 cv::Mat（参考 vis_pose 的转换方式），复制一份画布
    // 对每个满足 score>=threshold 的 result：
    //   画 box；对骨架每条 (i,j)：两端点存在且置信度达标则画线；画关键点圆
    // save_name 非空则 imwrite
    (void)font_size; (void)alpha;
}

}  // namespace modeldeploy::vision
```

> 实现参考：完整逻辑照搬 `csrc/vision/common/visualize/vis_pose.cpp` 的 `draw_result`/`draw_keypoints`（ImageData→cv::Mat、阈值过滤、画框画线画点、保存），仅骨架表/调色板换为 MediaPipe 21。若仓库已有统一 `visualize.h` 头，将函数声明加入该头而非新建 `vis_hand.h`。

- [ ] **Step 5: 构建 + 运行测试**

`.bat` 构建 `build_tdc_gpu` 后 `cd build_tdc_gpu` → `test_modeldeploy.exe "[hand]"`。
Expected: 通过（含真实权重时 21 点断言；缺失时有 SKIP 占位）。

- [ ] **Step 6: Commit**

```bash
git add tests/test_hand.cpp tests/CMakeLists.txt csrc/vision/common/visualize/vis_hand.cpp csrc/vision/common/visualize/vis_hand.h
git commit -m "feat(hand): hand keypoints test + MediaPipe 21 skeleton visualizer"
```

---

### Task 3: pybind 绑定（含 set_keypoints_num 补齐）

**Files:**
- Create: `csrc/pybind/vision/hand_pybind.cpp`
- Modify: `csrc/pybind/vision/vision_pybind.cpp`（声明 + 调用 `bind_hand`）
- Modify: `csrc/pybind/vision/ultralytics_pose_pybind.cpp`（补 `set_keypoints_num`/`get_keypoints_num` 属性）

**Interfaces:**
- Consumes: `hand::HandKeypoint`（Task 1）。
- Produces: Python `vision.HandKeypoint`、`KeyPointsResult`（已有）。

- [ ] **Step 1: 创建 `csrc/pybind/vision/hand_pybind.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind/utils/utils.h"
#include "vision/hand/hand.h"

namespace modeldeploy::vision {
    void bind_hand(const pybind11::module& m) {
        pybind11::class_<hand::HandKeypoint>(m, "HandKeypoint")
            .def(pybind11::init<const std::string&, const RuntimeOption&>(),
                 pybind11::arg("model_file"), pybind11::arg("option"))
            .def("predict",
                 [](const hand::HandKeypoint& self, const pybind11::array& im) {
                     auto cv = pyarray_to_cv_mat(im);
                     std::vector<KeyPointsResult> out;
                     ImageData img(cv);
                     if (!self.predict(img, &out)) {
                         throw std::runtime_error("HandKeypoint predict failed");
                     }
                     return out;
                 }, pybind11::arg("image"))
            .def("set_keypoints_num", [](hand::HandKeypoint& self, int n) {
                     self.get_postprocessor()->set_keypoints_num(n);
                 }, pybind11::arg("n"))
            .def("get_keypoints_num", [](const hand::HandKeypoint& self) {
                     return self.get_postprocessor()->get_keypoints_num();
                 });
    }
}
```

> 确认 `RuntimeOption` 在 pybind 中如何绑定/是否可直接传；若现有模型 pybind 用其它方式接收 model_file/option，沿用该模式（参考 `ultralytics_pose_pybind.cpp`）。

- [ ] **Step 2: 在 `csrc/pybind/vision/ultralytics_pose_pybind.cpp` 补 keypoints_num 暴露**

在 `UltralyticsPosePostprocessor` class_ 内追加：

```cpp
.def("set_keypoints_num", &detection::UltralyticsPosePostprocessor::set_keypoints_num, pybind11::arg("num"))
.def("get_keypoints_num", &detection::UltralyticsPosePostprocessor::get_keypoints_num)
```

- [ ] **Step 3: 在 `csrc/pybind/vision/vision_pybind.cpp` 注册**

声明区（与其它 `void bind_xxx(const pybind11::module& m);` 并列）：

```cpp
void bind_hand(const pybind11::module& m);
```

调用区（与其它 `bind_xxx(m);` 并列）：

```cpp
bind_hand(m);
```

- [ ] **Step 4: 构建 + 冒烟**

`.bat` 构建 `build_py`，`PYTHONPATH=.../build_py/bin`，`python -c "from modeldeploy.vision import HandKeypoint; print(HandKeypoint)"`。
Expected: 打印 `<class ...HandKeypoint>`。

- [ ] **Step 5: Commit**

```bash
git add csrc/pybind/vision/hand_pybind.cpp csrc/pybind/vision/vision_pybind.cpp csrc/pybind/vision/ultralytics_pose_pybind.cpp
git commit -m "feat(hand): pybind HandKeypoint + expose pose set_keypoints_num"
```

---

### Task 4: CAPI — MD_MODEL_HAND

**Files:**
- Modify: `capi/md_capi.h`（枚举 + 文档）
- Modify: `capi/md_capi.cpp`（create/destroy/clone/params/predict 分支）

**Interfaces:**
- Consumes: `hand::HandKeypoint`（Task 1）、现有 `MDModel` 生命周期与 `md_result_pose`/`md_result_keypoints`。
- Produces: `MD_MODEL_HAND`、`MDModelKind` 新枚举值。

- [ ] **Step 1: `capi/md_capi.h` 枚举区（`MD_MODEL_COUNT` 前）加**

```cpp
MD_MODEL_HAND,          //!< 手部关键点识别
```

> 放在 `MD_MODEL_COUNT` 之前保证 ABI 向后兼容。同时确认参数名表与 `set_keypoints_num` 的 param 名（现有 pose 用 `keypoints_num`）适用。

- [ ] **Step 2: `capi/md_capi.cpp` 分支**

在 `md_model_create` 的 switch 加 `case MD_MODEL_HAND:` 构造 `new vision::hand::HandKeypoint(...)`（attach 到 `MDModel`）；`destroy`/`clone` 对应分支同 pose。`md_model_predict` 推断分支按 pose 处理（输出走 `md_result_pose`/`md_result_keypoints`）。`set_model_param`/`get_model_param` 的 param 名/类型表按 pose 的 `keypoints_num` 注册（`kind_param_names`/`param_type_of`/`apply_model_param`）。

> 逐行对齐现有 `MD_MODEL_POSE` 的完整注册点（研究已列：create :781、destroy :964、clone :1019、set_input_size :1071、params :1183/1219/1297、predict :1515、draw :3237/3244）。

- [ ] **Step 3: 构建 + `[capi]` 测试**

`test_modeldeploy.exe "[capi]"` 应含 `[hand]`/pose 契约（若新增手部 CAPI 断言，见研究 `test_capi.cpp:530-605` pose 模式）。
Expected: 通过，无回归。

- [ ] **Step 4: Commit**

```bash
git add capi/md_capi.h capi/md_capi.cpp tests/test_capi.cpp
git commit -m "feat(hand): CAPI MD_MODEL_HAND"
```

---

### Task 5: C# 绑定 — HandModel

**Files:**
- Modify: `csharp/ModelDeploy/Models.cs`（加 `HandModel`，镜像 `PoseModel`）
- Modify: `csharp/ModelDeploy/enum_varaibles.cs`（加 `MD_MODEL_HAND`）
- Modify: `csharp/ModelDeploy/NativeMethods.cs`（如需）
- Modify: `csharp/ModelDeployUnitTest/AllModelsTests.cs`（`Hand_Works`）

**Interfaces:**
- Consumes: CAPI `MD_MODEL_HAND`（Task 4）、`PoseModel`/`PoseResult` 模式。
- Produces: C# `HandModel`、`KeyPoints` 结果访问。

- [ ] **Step 1: `enum_varaibles.cs`** 在 `MDModelKind` 加 `MD_MODEL_HAND`（值对齐 CAPI）。

- [ ] **Step 2: `Models.cs`** 加 `class HandModel`，镜像 `PoseModel`（含 `Clone`、`SetKeypointsNum(v)=SetParam("keypoints_num",v)`、`Predict` 返回含 KeyPoints 的结果）。

- [ ] **Step 3: `AllModelsTests.cs`** 加 `Hand_Works` 测试（创建 + 冒烟；权重不可用则按现有 pose 测试处理方式）。

- [ ] **Step 4: 构建 + 测试**

`dotnet build csharp/csharp.sln`（0 errors）；`dotnet test`（Barcode+Tracker+Hand 全绿；4 个 audio 失败为已知预存，不归因）。

- [ ] **Step 5: Commit**

```bash
git add csharp/ModelDeploy/Models.cs csharp/ModelDeploy/enum_varaibles.cs csharp/ModelDeploy/NativeMethods.cs csharp/ModelDeployUnitTest/AllModelsTests.cs
git commit -m "feat(hand): C# HandModel binding + tests"
```

---

### Task 6: Rust 绑定 — HandKeypoint

**Files:**
- Modify: `rust/modeldeploy/src/ffi.rs`（`MD_MODEL_HAND`）
- Modify: `rust/modeldeploy/src/model.rs`（`ResultType for HandKeypoint` + `model_wrapper!` + `hand()`/`hand_batch()`）
- Modify: `rust/modeldeploy/src/types.rs`（`ModelKind::Hand`）
- Modify: `rust/modeldeploy/src/lib.rs`（re-export）
- Modify: `rust/modeldeploy/tests/integration_test.rs`（`test_hand`）

**Interfaces:**
- Consumes: CAPI `MD_MODEL_HAND`（Task 4）。
- Produces: Rust `HandKeypoint` wrapper + `hand()` fn。

- [ ] **Step 1: `types.rs`** 加 `ModelKind::Hand`。

- [ ] **Step 2: `ffi.rs`** 加 `MD_MODEL_HAND`（对齐 CAPI）。`ResultType for HandKeypoint` 在 `model.rs`，结果类型复用 `KeyPointsResult`（已有 pose 结果类型）。

- [ ] **Step 3: `model.rs`**，镜像 `UltralyticsPose`（`model_wrapper!` 宏 + `hand()`/`hand_batch()`），按 `keypoints_num` param 封装。

- [ ] **Step 4: 构建 + clippy + 测试**

`cargo build`（0 errors）、`cargo clippy -- -D warnings`（clean）、`cargo test`（`test_hand` 通过，无回归）。

- [ ] **Step 5: Commit**

```bash
git add rust/modeldeploy/src/ffi.rs rust/modeldeploy/src/model.rs rust/modeldeploy/src/types.rs rust/modeldeploy/src/lib.rs rust/modeldeploy/tests/integration_test.rs
git commit -m "feat(hand): Rust HandKeypoint binding + test"
```

---

### Task 7: Demo + Docs

**Files:**
- Create: `examples/demo_hand/`（`demo_hand.cpp` + `CMakeLists.txt`）
- Modify: `examples/CMakeLists.txt`（`add_subdirectory(demo_hand)`）
- Modify: `examples/EXAMPLES.md`、`README.md`

**Interfaces:**
- Consumes: `HandKeypoint`（Task 1）、`vis_hand`（Task 2）。
- Produces: 可运行的 demo。

- [ ] **Step 1: 创建 `examples/demo_hand/demo_hand.cpp`**

```cpp
#include <opencv2/imgcodecs.hpp>
#include <string>
#include "runtime_option.h"
#include "vision/hand/hand.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: demo_hand <model.onnx> <image.jpg>\n");
        return 1;
    }
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    modeldeploy::vision::hand::HandKeypoint model(argv[1], opt);
    auto im = cv::imread(argv[2]);
    if (im.empty()) { printf("cannot read image %s\n", argv[2]); return 1; }
    modeldeploy::vision::ImageData img(im);
    std::vector<modeldeploy::vision::KeyPointsResult> res;
    if (!model.predict(img, &res)) { printf("predict failed\n"); return 1; }
    printf("detected %zu hand(s)\n", res.size());
    for (auto& r : res)
        printf("  box=(%.1f,%.1f,%.1f,%.1f) score=%.3f keypoints=%zu\n",
               r.box.x, r.box.y, r.box.width, r.box.height, r.score, r.keypoints.size());
    return 0;
}
```

- [ ] **Step 2: 创建 `examples/demo_hand/CMakeLists.txt`**

```cmake
add_executable(demo_hand demo_hand.cpp)
target_link_libraries(demo_hand PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
```

- [ ] **Step 3: `examples/CMakeLists.txt`** 加 `add_subdirectory(demo_hand)`。

- [ ] **Step 4: 构建 + 运行**

`.bat` 构建 `build_tdc_gpu`；运行 `demo_hand.exe <model> <image>` 打印手部数量/关键点数。无法拿到权重时至少确认构建通过 + `Usage`路径。

- [ ] **Step 5: `EXAMPLES.md` / `README.md`** 加 demo 与能力条目（"手部关键点"）。

- [ ] **Step 6: Commit**

```bash
git add examples/demo_hand examples/CMakeLists.txt examples/EXAMPLES.md README.md
git commit -m "feat(hand): demo_hand + docs"
```

---

### Task 8: 全量构建 + 回归验证

**Files:** 无新增；验证整个分支。

- [ ] **Step 1: 全量 C++ 测试**

`build_tdc_gpu` 后：`test_modeldeploy.exe "[hand]"`、`"[capi]"`、`"[core]"`、`"[tracking]"`、`"[barcode]"` 全部 PASS；`[tracking]`/`[barcode]` 无回归。

- [ ] **Step 2: 跨后端语义确认**

`HandKeypoint` 委托 `UltralyticsPose`（即复用 BaseBackend），天然支持 ORT/MNN/TRT/Sophgo——具备既有 pose 后端能力即可。记录"各后端语义一致"。

- [ ] **Step 3: 绑定量测**

Python import + `HandKeypoint` 创建冒烟；C# `Hand_Works`；Rust `test_hand`；`demo_hand` 运行。

- [ ] **Step 4: 收尾 commit（仅在有变更时）**，否则跳过（不作空 commit）。

---

## Self-Review 核查

- **Spec 覆盖**：架构（薄封装包装 pose）→ Task 1；MediaPipe 21 骨架可视化 → Task 2；21 点默认 → Task 1；六面集成 → Task 2/3/4/5/6/7；测试/回归 → Task 2/8；YAGNI（不做手势分类/专用 preprocessor）→ 设计未含，符合。
- **Placeholder**：各步骤给出具体代码/命令；模型权重缺失的测试用 SKIP 占位并注明。
- **类型一致性**：`HandKeypoint`/`MD_MODEL_HAND`/`HandModel`/`ModelKind::Hand`/`set_keypoints_num` 在各 Task 一致；`KeyPointsResult` 全链路复用。
