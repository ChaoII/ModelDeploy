# Item 8: 面部 Landmark / 车辆关键点（关键点扩展）——实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在既有 `detection::UltralyticsPose`（姿态 17 点）与 `hand::HandKeypoint`（手部 21 点）关键点家族之上，新增两个独立关键点模型——**`landmark::VehicleKeypoint`**（车辆关键点，薄封装 `UltralyticsPose` 复用 `set_keypoints_num` 泛化）与 **`landmark::FaceLandmark`**（面部 106 点，走 spec §12 专用实现出口复用已有 `face::InsightFaceLandmark`），6 面贯通按能力降级（YAGNI）。

**Architecture:** `VehicleKeypoint` 逐字对齐 `hand::HandKeypoint` —— 内含 `pose_`（by-value）薄封装，构造用 `get_postprocessor().set_keypoints_num(4)`，输出统一 `KeyPointsResult`。`FaceLandmark` 因 insightface 106 模型为"仿射对齐裁剪 + 逆仿射回投"预处理/后处理（与 pose 的 letterbox 管线不兼容，见任务 2 说明），故内部复用已建的 `face::InsightFaceLandmark`（不重建解析），`predict(人脸裁剪图)` 把 2d106 输出的 106 个 `array<float,2>` 适配为 `KeyPointsResult`。两模型输出均为 `KeyPointsResult` → CAPI 直接复用 `MD_RES_POSE` + `md_result_keypoints`（与 HAND 完全相同），C#/Rust 复用 `PoseResult`/`Pose` 读取。可视化用既有 `vis_keypoints`。

**Tech Stack:** C++17、pybind11、Catch2、OpenCV、现有 `BaseModel`/`Runtime`/`Tensor`/`ImageData`/`UltralyticsPose`/`InsightFaceLandmark`/`KeyPointsResult`。

**Spec:** `docs/superpowers/specs/2026-08-22-keypoint-expansion-design.md`

## Global Constraints

- **命名空间与目录**：`modeldeploy::vision::landmark`；新文件放 `csrc/vision/landmark/`（`face_landmark.{h,cpp}`、`vehicle_keypoint.{h,cpp}`），被根 `CMakeLists.txt` 的 `file(GLOB_RECURSE ... csrc/*.cpp)` 自动收集，**无需改主 CMake**（纯 C++ + 既有 vision 依赖）。
- **复用 `UltralyticsPose` 泛化**：`UltralyticsPose` 在 `modeldeploy::vision::detection` 命名空间（`csrc/vision/pose/ultralytics_pose.h:13`，注意**不是** `vision::pose` 命名空间）。`set_keypoints_num`/`get_keypoints_num` 位于 `UltralyticsPosePostprocessor`（`csrc/vision/pose/postprocessor.h:46`），经 `get_postprocessor()` 取得——`set_keypoints_num` 参数是 `int`，`get_keypoints_num` 返回 `float`（既有怪癖，照抄 HAND 即可）。
- **`HandKeypoint` 是面对面模板（逐字对照）**：`csrc/vision/hand/hand.{h,cpp}` 内含 by-value `pose_` **薄封装**（**不是** `BaseModel` 子类），暴露 `predict/batch_predict/draw_result/is_initialized/clone/get_preprocessor/get_postprocessor`。**设计规范 §3 提案的 `BaseModel` 子类 + unique_ptr 形式与现有先例不符**，本计划遵循实际先例（spec §12 已预留按实现调整的余地）。
- **结果结构不复用**：`KeyPointsResult`（`csrc/vision/common/result.h:80`）含 `{box: Rect2f, keypoints: vector<Point3f>, label_id, score}`；`Point3f`/`Rect2f` 在 `csrc/vision/common/struct.h`。可视化 `vis_keypoints`（`csrc/vision/common/visualize/visualize.h:56`）、`vis_hand`（:62）。不新增结果结构。
- **CAPI 结果复用**：两模型输出 `std::vector<KeyPointsResult>` → CAPI 结果句柄 kind 直接用 `MD_RES_POSE`，读取走 `md_result_pose` + `md_result_keypoints`，**与 `MD_MODEL_HAND` 完全一致**（`capi/md_capi.cpp:1592/1881`）。不新增结果枚举。
- **无权重 → 测试 SKIP**：模型 construct（`is_initialized()==false`）、`[landmark]` 单测对真实权重外链缺失 SKIP（仿 `tests/test_hand.cpp`，`TEST_DATA_DIR` 环境变量定位）。
- **权重外链 SKIP**：vehicle/face 模型权重均在 modelscope 外链，仓库内不包含；测试/demo 缺权重时优雅报错不崩溃。
- **MSVC `/utf-8`**：根 CMake 已为 SDK 自动设置；C++17。
- **`FaceLandmark` 不接入 set_param/set_size**：无 pose postprocessor，不注册 `keypoints_num/conf_threshold/nms_threshold`（CAPI 对其 set_param 走 default 返回 unsupported）。`VehicleKeypoint` 完整镜像 HAND（注册 conf/nms/keypoints_num）。

---

### Task 1: `VehicleKeypoint` C++ 核心类 + 单测

**Files:**
- Create: `csrc/vision/landmark/vehicle_keypoint.h`
- Create: `csrc/vision/landmark/vehicle_keypoint.cpp`
- Test: `tests/test_landmark.cpp`（本 Task 只含 VehicleKeypoint 用例）
- Modify: `tests/CMakeLists.txt`（TEST_SOURCES 加 `test_landmark.cpp`）

**Interfaces:**
- Produces (later tasks rely on):
  - `class modeldeploy::vision::landmark::VehicleKeypoint`
  - `VehicleKeypoint(const std::string& model_file, const RuntimeOption& option = RuntimeOption())`
  - `bool predict(const ImageData& img, std::vector<KeyPointsResult>* results, TimerArray* timer = nullptr)`
  - `bool batch_predict(const std::vector<ImageData>& imgs, std::vector<std::vector<KeyPointsResult>>* results, TimerArray* timer = nullptr)`
  - `bool draw_result(ImageData& img, const std::vector<KeyPointsResult>& results, double threshold = 0.5)`
  - `bool is_initialized() const`；`std::unique_ptr<VehicleKeypoint> clone() const`
  - `UltralyticsPosePreprocessor& get_preprocessor()`；`UltralyticsPosePostprocessor& get_postprocessor()`
  - 构造后 `pose_` 的 postprocessor `keypoints_num` 默认 = **4**

- [ ] **Step 1: 写失败测试** (`tests/test_landmark.cpp`，含 `[landmark]` 标签)

```cpp
#include "catch2/catch_test_macros.hpp"
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "vision/landmark/vehicle_keypoint.h"

namespace fs = std::filesystem;
using namespace modeldeploy::vision;

namespace {
    fs::path vehicle_model_path() {
        const char* dir = std::getenv("TEST_DATA_DIR");
        const fs::path base = (dir && *dir) ? fs::path(dir) / "test_data" : fs::current_path() / "test_data";
        return base / "test_models" / "onnx" / "vehicle_keypoint.onnx";
    }
}

TEST_CASE("VehicleKeypoint defaults to 4 keypoints", "[landmark]") {
    auto modelfile = vehicle_model_path();
    if (!fs::exists(modelfile)) {
        WARN("vehicle_keypoint.onnx 权重缺失（外链 modelscope），跳过默认 4 点断言");
        return;
    }
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    landmark::VehicleKeypoint model(modelfile.string(), opt);
    REQUIRE(model.get_postprocessor().get_keypoints_num() == 4);
    REQUIRE(model.is_initialized());
}

TEST_CASE("VehicleKeypoint set_keypoints_num override", "[landmark]") {
    auto modelfile = vehicle_model_path();
    if (!fs::exists(modelfile)) {
        WARN("vehicle_keypoint.onnx 权重缺失，跳过 set_keypoints_num 断言");
        return;
    }
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    landmark::VehicleKeypoint model(modelfile.string(), opt);
    model.get_postprocessor().set_keypoints_num(8);
    REQUIRE(model.get_postprocessor().get_keypoints_num() == 8);
}

TEST_CASE("VehicleKeypoint construction without weights", "[landmark]") {
    landmark::VehicleKeypoint model("nonexistent_vehicle_keypoint.onnx");
    REQUIRE_FALSE(model.is_initialized());
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[landmark]"`
Expected: FAIL（`csrc/vision/landmark/vehicle_keypoint.h` 不存在，编译失败）
> `build` 目录需先配置：`cmake -S . -B build -G Ninja -DBUILD_AUDIO=ON -DBUILD_VISION=ON -DBUILD_CAPI=ON -DBUILD_PYTHON=OFF -DENABLE_MNN=OFF -DENABLE_ORT=ON -DENABLE_TRT=OFF -DWITH_GPU=OFF -DBUILD_TESTS=ON`。并先向 `tests/CMakeLists.txt` 的 `TEST_SOURCES` 加 `test_landmark.cpp`（Step 8 已含）。

- [ ] **Step 3: 写 `csrc/vision/landmark/vehicle_keypoint.h`**

```cpp
#pragma once

#include <memory>
#include <string>

#include "base_model.h"
#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/pose/ultralytics_pose.h"

namespace modeldeploy::vision::landmark {
    /*! @brief 车辆关键点识别（默认 4 车轮关键点），薄封装 UltralyticsPose（复用 set_keypoints_num 泛化）。 */
    class MODELDEPLOY_CXX_EXPORT VehicleKeypoint {
    public:
        explicit VehicleKeypoint(const std::string& model_file,
                                 const RuntimeOption& option = RuntimeOption());

        bool predict(const ImageData& img,
                     std::vector<KeyPointsResult>* results,
                     TimerArray* timer = nullptr);

        bool batch_predict(const std::vector<ImageData>& imgs,
                           std::vector<std::vector<KeyPointsResult>>* results,
                           TimerArray* timer = nullptr);

        bool draw_result(ImageData& img,
                         const std::vector<KeyPointsResult>& results,
                         double threshold = 0.5);

        vision::detection::UltralyticsPosePreprocessor& get_preprocessor() {
            return pose_.get_preprocessor();
        }

        vision::detection::UltralyticsPosePostprocessor& get_postprocessor() {
            return pose_.get_postprocessor();
        }

        bool is_initialized() const { return pose_.is_initialized(); }

        std::unique_ptr<VehicleKeypoint> clone() const;

    private:
        vision::detection::UltralyticsPose pose_;
    };
} // namespace modeldeploy::vision::landmark
```

- [ ] **Step 4: 写 `csrc/vision/landmark/vehicle_keypoint.cpp`**

```cpp
#include "vision/landmark/vehicle_keypoint.h"
#include "vision/common/visualize/visualize.h"

namespace modeldeploy::vision::landmark {
    VehicleKeypoint::VehicleKeypoint(const std::string& model_file, const RuntimeOption& option)
        : pose_(model_file, option) {
        // 默认 4 车轮关键点；不同车型模型可用 set_keypoints_num 覆盖（spec §12 已强调点数泛化）。
        pose_.get_postprocessor().set_keypoints_num(4);
    }

    bool VehicleKeypoint::predict(const ImageData& img,
                                  std::vector<KeyPointsResult>* results,
                                  TimerArray* timer) {
        return pose_.predict(img, results, timer);
    }

    bool VehicleKeypoint::batch_predict(const std::vector<ImageData>& imgs,
                                        std::vector<std::vector<KeyPointsResult>>* results,
                                        TimerArray* timer) {
        return pose_.batch_predict(imgs, results, timer);
    }

    bool VehicleKeypoint::draw_result(ImageData& img,
                                      const std::vector<KeyPointsResult>& results,
                                      double threshold) {
        (void)threshold;
        img = vis_keypoints(img, results, "", 14, 4, 0.15, false);
        return true;
    }

    std::unique_ptr<VehicleKeypoint> VehicleKeypoint::clone() const {
        auto ret = std::make_unique<VehicleKeypoint>(*this);
        ret->pose_.set_runtime(ret->pose_.clone_runtime());
        return ret;
    }
} // namespace modeldeploy::vision::landmark
```

> **实现提示**：`vis_keypoints` 返回 `ImageData`（`visualize.h:56`）——若签名与 `vis_hand` 有细节差异，以实参顺序 `(image, result, text, line_width, radius, & 由实现定)` 为准校正（`vis_hand(img, results, "", 14, 4, 0.15, false)` 为在册例子）。`UltralyticsPose::batch_predict` 输出为 `vector<vector<KeyPointsResult>>`；本类透传。

- [ ] **Step 5: 写测试注册 + 构建 + 通过**

Modify `tests/CMakeLists.txt` TEST_SOURCES：加 `test_landmark.cpp`（在 `test_hand.cpp` 之后）。
Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[landmark]"`
Expected: 3 个 `[landmark]` 用例 PASS（无权重 SKIP 分支 WARN 通过）。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/landmark/vehicle_keypoint.h csrc/vision/landmark/vehicle_keypoint.cpp tests/test_landmark.cpp tests/CMakeLists.txt
git commit -m "feat(landmark): VehicleKeypoint thin wrapper over UltralyticsPose + tests"
```

---

### Task 2: `FaceLandmark` C++ 核心类 + 单测

> **关键决策（Spec §12 出口）**：insightface 2d106 模型**不可直接复用 `UltralyticsPose` 后端**——它走"bbox→仿射对齐裁剪（affine）→推理→逆仿射回投"（`csrc/vision/face/insightface/landmark/insightface_landmark.{h,cpp}`），后处理输出 `(1,106,2)` 原始坐标，与 pose 的 letterbox + YOLO `[1,(5+2k)*A]` 输出**格式不兼容**。因此 `FaceLandmark` 走 spec §12 专用实现出口：**内部复用已建的 `face::InsightFaceLandmark`（不重建解析），`predict(人脸裁剪图)` 以整图为 bbox 适配 2d106 输出到 `KeyPointsResult`**。这样 public `predict(ImageData, KeyPointsResult*)` 与 `HandKeypoint`/`VehicleKeypoint` 统一，CAPI/C#/Rust 全链复用 pose 结果读取。

**Files:**
- Create: `csrc/vision/landmark/face_landmark.h`
- Create: `csrc/vision/landmark/face_landmark.cpp`
- Modify: `tests/test_landmark.cpp`（追加 FaceLandmark 用例）

**Interfaces:**
- Consumes: `face::InsightFaceLandmark` (`csrc/vision/face/insightface/landmark/insightface_landmark.h`)，`predict_2d106(image, bbox, vector<array<float,2>>* landmarks, TimerArray*)`
- Produces (later tasks rely on):
  - `class modeldeploy::vision::landmark::FaceLandmark`
  - `FaceLandmark(const std::string& model_file, const RuntimeOption& option = RuntimeOption())`
  - `bool predict(const ImageData& img, std::vector<KeyPointsResult>* results, TimerArray* timer = nullptr)` —— 对裁剪图输出 106 点
  - `bool is_initialized() const`；`std::unique_ptr<FaceLandmark> clone() const`

- [ ] **Step 1: 追加失败测试** (`tests/test_landmark.cpp`)

```cpp
#include "vision/landmark/face_landmark.h"

namespace {
    fs::path face_model_path() {
        const char* dir = std::getenv("TEST_DATA_DIR");
        const fs::path base = (dir && *dir) ? fs::path(dir) / "test_data" : fs::current_path() / "test_data";
        return base / "test_models" / "onnx" / "2d106det.onnx";
    }
}

TEST_CASE("FaceLandmark construction without weights", "[landmark]") {
    landmark::FaceLandmark model("nonexistent_2d106det.onnx");
    REQUIRE_FALSE(model.is_initialized());
}

TEST_CASE("FaceLandmark predict produces 106 keypoints", "[landmark]") {
    auto modelfile = face_model_path();
    if (!fs::exists(modelfile)) {
        WARN("2d106det.onnx 权重缺失（外链 modelscope），跳过 predict 主路径");
        return;
    }
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    landmark::FaceLandmark model(modelfile.string(), opt);

    cv::Mat canvas(192, 192, CV_8UC3, cv::Scalar(128, 128, 128));
    ImageData img(canvas);
    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results));
    REQUIRE_FALSE(results.empty());
    REQUIRE(results.front().keypoints.size() == 106);
    REQUIRE(results.front().box.width > 0);
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[landmark]"`
Expected: FAIL（`csrc/vision/landmark/face_landmark.h` 不存在，编译失败）

- [ ] **Step 3: 写 `csrc/vision/landmark/face_landmark.h`**

```cpp
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/face/insightface/landmark/insightface_landmark.h"

namespace modeldeploy::vision::landmark {
    /*! @brief 面部 Landmark 独立访问（InsightFace 2d106，106 点）。
     *  输入：人脸裁剪图；输出：KeyPointsResult（106 个 Point3f，z=0）。
     *  insightface 106 模型为仿射对齐预/后处理，与 pose letterbox 后端不兼容，
     *  故走 spec §12 专用实现出口：内部复用已建的 face::InsightFaceLandmark，仅做薄适配。
     */
    class MODELDEPLOY_CXX_EXPORT FaceLandmark {
    public:
        explicit FaceLandmark(const std::string& model_file,
                              const RuntimeOption& option = RuntimeOption());

        bool predict(const ImageData& img,
                     std::vector<KeyPointsResult>* results,
                     TimerArray* timer = nullptr);

        bool is_initialized() const;

        std::unique_ptr<FaceLandmark> clone() const;

    private:
        explicit FaceLandmark(std::unique_ptr<face::InsightFaceLandmark> lm);
        std::unique_ptr<face::InsightFaceLandmark> landmark_;
    };
} // namespace modeldeploy::vision::landmark
```

- [ ] **Step 4: 写 `csrc/vision/landmark/face_landmark.cpp`**

```cpp
#include "vision/landmark/face_landmark.h"

namespace modeldeploy::vision::landmark {
    FaceLandmark::FaceLandmark(const std::string& model_file, const RuntimeOption& option)
        : landmark_(std::make_unique<face::InsightFaceLandmark>(model_file, option)) {}

    FaceLandmark::FaceLandmark(std::unique_ptr<face::InsightFaceLandmark> lm)
        : landmark_(std::move(lm)) {}

    bool FaceLandmark::is_initialized() const {
        return landmark_ && landmark_->is_initialized();
    }

    bool FaceLandmark::predict(const ImageData& img,
                               std::vector<KeyPointsResult>* results,
                               TimerArray* timer) {
        if (!results || !landmark_) return false;
        // 对"人脸裁剪图"整体估计 106 点：bbox 取整图范围（裁剪图输入约定，spec §3.2）。
        const float W = static_cast<float>(img.width());
        const float H = static_cast<float>(img.height());
        std::array<float, 4> bbox{0.f, 0.f, W, H};
        std::vector<std::array<float, 2>> lms;
        if (!landmark_->predict_2d106(img, bbox, &lms, timer)) return false;

        KeyPointsResult r;
        r.box = Rect2f(0.f, 0.f, W, H);
        r.label_id = 0;
        r.score = 1.0f;
        r.keypoints.reserve(lms.size());
        for (const auto& p : lms)
            r.keypoints.emplace_back(p[0], p[1], 0.f);
        results->clear();
        results->push_back(std::move(r));
        return true;
    }

    std::unique_ptr<FaceLandmark> FaceLandmark::clone() const {
        // InsightFaceLandmark::clone() 会基于存储的 model_file + runtime 重建独立实例
        auto lm = landmark_ ? landmark_->clone() : nullptr;
        return std::make_unique<FaceLandmark>(std::move(lm));
    }
} // namespace modeldeploy::vision::landmark
```

> **实现提示**：`InsightFaceLandmark::clone()`（`insightface_landmark.cpp:169`）从 `runtime_option.model_file` 重建，语义正确。若 `predict_2d106` 因输入尺寸/通道断言失败，把输入预处理为 insightface 需要的大小（其内部 preprocessor 已按 `input_size_{192,192}` 处理）——测试以 192×192 画布规避该分支；真实权重联调（Task 8）校正。

- [ ] **Step 5: 构建 + 运行时确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[landmark]"`
Expected: 5 个 `[landmark]` 用例 PASS（VehicleKeypoint 3 + FaceLandmark 2）。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/landmark/face_landmark.h csrc/vision/landmark/face_landmark.cpp tests/test_landmark.cpp
git commit -m "feat(landmark): FaceLandmark thin adapter over InsightFaceLandmark (2d106) + tests"
```

---

### Task 3: pybind（`modeldeploy.vision.landmark.VehicleKeypoint` / `FaceLandmark`）

**Files:**
- Create: `csrc/pybind/vision/landmark_pybind.cpp`
- Modify: `csrc/pybind/vision/vision_pybind.cpp`（前置声明 + 注册 `bind_landmark`）
- Test: （Python smoke）

**Interfaces:**
- Consumes: `VehicleKeypoint`/`FaceLandmark` (Task 1/2)、`ImageData` 绑定（vision 子模块已注册 `image_data_pybind.cpp`）
- Produces: Python `modeldeploy.vision.landmark.VehicleKeypoint` / `FaceLandmark`，`predict(im) -> List[KeyPointsResult]`、`set/get_keypoints_num`（仅 Vehicle）

- [ ] **Step 1: 写 `csrc/pybind/vision/landmark_pybind.cpp`**（镜像 `hand_pybind.cpp`）

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind/utils/utils.h"
#include "vision/landmark/face_landmark.h"
#include "vision/landmark/vehicle_keypoint.h"

namespace modeldeploy::vision {
    void bind_landmark(const pybind11::module& m) {
        pybind11::class_<landmark::VehicleKeypoint>(m, "VehicleKeypoint")
            .def(pybind11::init<const std::string&, const RuntimeOption&>(),
                 pybind11::arg("model_file"), pybind11::arg("option"))
            .def("predict",
                 [](landmark::VehicleKeypoint& self, const pybind11::array& im) {
                     auto cv = pyarray_to_cv_mat(im);
                     std::vector<KeyPointsResult> out;
                     ImageData img(cv);
                     if (!self.predict(img, &out))
                         throw std::runtime_error("VehicleKeypoint predict failed");
                     return out;
                 }, pybind11::arg("image"))
            .def("set_keypoints_num",
                 [](landmark::VehicleKeypoint& self, int n) {
                     self.get_postprocessor().set_keypoints_num(n);
                 }, pybind11::arg("n"))
            .def("get_keypoints_num",
                 [](landmark::VehicleKeypoint& self) {
                     return self.get_postprocessor().get_keypoints_num();
                 })
            .def("is_initialized", &landmark::VehicleKeypoint::is_initialized);

        pybind11::class_<landmark::FaceLandmark>(m, "FaceLandmark")
            .def(pybind11::init<const std::string&, const RuntimeOption&>(),
                 pybind11::arg("model_file"), pybind11::arg("option"))
            .def("predict",
                 [](landmark::FaceLandmark& self, const pybind11::array& im) {
                     auto cv = pyarray_to_cv_mat(im);
                     std::vector<KeyPointsResult> out;
                     ImageData img(cv);
                     if (!self.predict(img, &out))
                         throw std::runtime_error("FaceLandmark predict failed");
                     return out;
                 }, pybind11::arg("image"))
            .def("is_initialized", &landmark::FaceLandmark::is_initialized);
    }
} // namespace modeldeploy::vision
```

- [ ] **Step 2: 编辑 `csrc/pybind/vision/vision_pybind.cpp`**

在声明区加 `void bind_landmark(pybind11::module&);`（紧邻 `bind_action` 声明），并在 `bind_vision` 内 `bind_action(m);` 之后加 `bind_landmark(m);`（该文件整体在 vision 子模块 `#ifdef BUILD_VISION` 作用域内）。`FaceLandmark` include 了 insightface 头，复用 vision 子模块已注册的 `ImageData` 绑定。

- [ ] **Step 3: 构建 + Python smoke**

用开 `BUILD_VISION=ON BUILD_PYTHON=ON` 的构建（如 `build_py`）。构建后：
```bash
cd build_py && python -c "
import modeldeploy
from modeldeploy.vision import landmark
v = landmark.VehicleKeypoint('nonexistent.onnx')
assert v.is_initialized() == False
# 默认 4 点（构造即设，不依赖权重）
assert v.get_keypoints_num() == 4
v.set_keypoints_num(8)
assert v.get_keypoints_num() == 8
f = landmark.FaceLandmark('nonexistent_2d106.onnx')
assert f.is_initialized() == False
print('landmark smoke OK')
"
```
Expected: `landmark smoke OK`，无异常。

- [ ] **Step 4: Commit**

```bash
git add csrc/pybind/vision/landmark_pybind.cpp csrc/pybind/vision/vision_pybind.cpp
git commit -m "feat(pybind): bind vision.landmark VehicleKeypoint + FaceLandmark"
```

---

### Task 4: CAPI（`MD_MODEL_VEHICLE_KEYPOINT` / `MD_MODEL_FACE_LANDMARK` + pose 结果复用）

**Files:**
- Modify: `capi/md_capi.h`（枚举 `MD_MODEL_VEHICLE_KEYPOINT`/`MD_MODEL_FACE_LANDMARK`）
- Modify: `capi/md_capi.cpp`（create/delete/clone/set_size/set_param/kind_param_names/param_type_of/predict/batch——Vehicle 全镜像 HAND；Face 仅 create/delete/clone/predict/batch）
- Test: `tests/test_capi.cpp`（`[capi]` 用例）

**Interfaces:**
- Consumes: `VehicleKeypoint`/`FaceLandmark` (Task 1/2)
- Produces: C 枚举 `MD_MODEL_VEHICLE_KEYPOINT`/`MD_MODEL_FACE_LANDMARK`（`MD_MODEL_COUNT` 前）；二者 `md_model_create`/`md_model_predict`/`md_model_predict_batch` 结果 kind=`MD_RES_POSE`，用既有 `md_result_pose`+`md_result_keypoints` 读取。Vehicle 支持 `set_param("keypoints_num"|"conf_threshold"|"nms_threshold")` + `md_model_set_input_size`；Face 不支持 set_param/set_size。

- [ ] **Step 1: `capi/md_capi.h` 枚举** — 在 `MD_MODEL_HAND` 附近、`MD_MODEL_COUNT` 之前加：
```c
MD_MODEL_VEHICLE_KEYPOINT,
MD_MODEL_FACE_LANDMARK,
```

- [ ] **Step 2: `capi/md_capi.cpp` 分发**（镜像 HAND，`#ifdef BUILD_VISION` 分支）：
- create（仿 `:791` HAND）：
```cpp
case MD_MODEL_VEHICLE_KEYPOINT: {
    mh->model = make_model<vision::landmark::VehicleKeypoint>(model_path, opt, "VehicleKeypoint", &err);
    if (!mh->model) return fail_init("VehicleKeypoint");
    break;
}
case MD_MODEL_FACE_LANDMARK: {
    mh->model = make_model<vision::landmark::FaceLandmark>(model_path, opt, "FaceLandmark", &err);
    if (!mh->model) return fail_init("FaceLandmark");
    break;
}
```
- delete（仿 `:1013`）：`case MD_MODEL_VEHICLE_KEYPOINT: delete static_cast<vision::landmark::VehicleKeypoint*>(model); break;`；`MD_MODEL_FACE_LANDMARK` 同理（`FaceLandmark`）。
- clone（仿 `:1074`）：`case MD_MODEL_VEHICLE_KEYPOINT: cloned = static_cast<vision::landmark::VehicleKeypoint*>(src->model)->clone().release(); break;`；Face 同理。
- set_size（仿 `:1132`）：**仅 Vehicle**：`case MD_MODEL_VEHICLE_KEYPOINT: static_cast<vision::landmark::VehicleKeypoint*>(mh->model)->get_preprocessor().set_size(size); break;`
- `kind_param_names`（`md_capi.cpp:1244`）：`case MD_MODEL_POSE: case MD_MODEL_HAND: case MD_MODEL_VEHICLE_KEYPOINT:` → `"conf_threshold|nms_threshold|keypoints_num"`。
- `param_type_of`（`md_capi.cpp:1277`）：`case MD_MODEL_POSE: case MD_MODEL_HAND: case MD_MODEL_VEHICLE_KEYPOINT:`（并在 `:1282` 的 `keypoints_num` 判断加 `|| kind == MD_MODEL_VEHICLE_KEYPOINT`）。
- set_param 分发（仿 `:1367` HAND）：**仅 Vehicle**：
```cpp
case MD_MODEL_VEHICLE_KEYPOINT: {
    auto* pm = static_cast<vision::landmark::VehicleKeypoint*>(const_cast<void*>(m));
    if (std::strcmp(name, "conf_threshold") == 0) pm->get_postprocessor().set_conf_threshold((float)d);
    else if (std::strcmp(name, "nms_threshold") == 0) pm->get_postprocessor().set_nms_threshold((float)d);
    else pm->get_postprocessor().set_keypoints_num((int)i);
    break;
}
```
（`MD_MODEL_FACE_LANDMARK` 不在此处——set_param 走 default 返回 unsupported。）
- predict 单图（仿 `:1592`）：
```cpp
case MD_MODEL_VEHICLE_KEYPOINT: {
    auto* m = static_cast<vision::landmark::VehicleKeypoint*>(mh->model);
    auto* d = new ResultData<KeyPointsResult>();
    if (!m->predict(image, &d->v)) return predict_fail("vehicle_keypoint");
    rh->kind = MD_RES_POSE; rh->data = d; break;
}
case MD_MODEL_FACE_LANDMARK: {
    auto* m = static_cast<vision::landmark::FaceLandmark*>(mh->model);
    auto* d = new ResultData<KeyPointsResult>();
    if (!m->predict(image, &d->v)) return predict_fail("face_landmark");
    rh->kind = MD_RES_POSE; rh->data = d; break;
}
```
- predict batch（仿 `:1881`）：`VehicleKeypoint`/`FaceLandmark` 各 `new ResultData<std::vector<KeyPointsResult>>()` 逐图 `m->predict(image_at(i), &r)`，`rh->kind = MD_RES_POSE`。

> **实现提示**：务必逐字复制 HAND 的结果装填模式（`ResultData<KeyPointsResult>` + `MD_RES_POSE`），保证 `md_result_keypoints` 读取语义一致。若 `md_model_set_input_size` 的 set_size 分发在别处还有 `get_size` 类开关，镜像 HAND 一并处理。

- [ ] **Step 3: `tests/test_capi.cpp`** — 新增 `[capi]` 用例：枚举 `MD_MODEL_VEHICLE_KEYPOINT`/`MD_MODEL_FACE_LANDMARK` 创建（模型缺失 guard 跳过）；`md_model_predict` 空 image/空 handle → `MD_ERROR_INVALID`；Vehicle `md_model_set_param_i(...,"keypoints_num",8)` 后 `get_postprocessor().get_keypoints_num()==8`（经 set_param_i，模型缺失 SKIP）。

- [ ] **Step 4: 构建 + 测试**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[capi]"`
Expected: 0 errors；landmark 用例（模型缺失 SKIP，错误路径通过）。

- [ ] **Step 5: Commit**

```bash
git add capi/md_capi.h capi/md_capi.cpp tests/test_capi.cpp
git commit -m "feat(capi): MD_MODEL_VEHICLE_KEYPOINT/FACE_LANDMARK reusing MD_RES_POSE"
```

---

### Task 5: C#（`VehicleKeypointModel` / `FaceLandmarkModel` 薄封装）

> **降级判断（推荐保留）**：C# 有 `HandModel`（`csharp/ModelDeploy/Models.cs:174`）与 `PoseResult` 读取全链现成模板，新增两模型仅"复制改名 + 换枚举"，成本极低、价值高。**建议保留**；若时间紧可仅交付 Vehicle 而将 FaceLandmark 标注 DEV-SKIPPED（C++/pybind/CAPI 已交付）。

**Files:**
- Modify: `csharp/ModelDeploy/types_internal_c.cs`（`MDModelKind` 加 `MD_MODEL_VEHICLE_KEYPOINT`/`MD_MODEL_FACE_LANDMARK`）
- Modify: `csharp/ModelDeploy/Models.cs`（仿 `HandModel` 加两模型）
- Test: `csharp/ModelDeployUnitTest/AllModelsTests.cs`（`VehicleKeypoint_Works` / `FaceLandmark_Works`）

**Interfaces:**
- Consumes: CAPI (Task 4)
- Produces: C# `ModelDeploy.VehicleKeypointModel` / `FaceLandmarkModel`，`Predict(VisionImage) -> Prediction<PoseResult[]>`、`SetKeypointsNum`（仅 Vehicle）、`Clone`。

- [ ] **Step 1**: `types_internal_c.cs` `MDModelKind` 加 `MD_MODEL_VEHICLE_KEYPOINT,` / `MD_MODEL_FACE_LANDMARK,`（`MD_MODEL_HAND` 之后，值对齐 CAPI 枚举顺序）。
- [ ] **Step 2**: `Models.cs` 仿 `HandModel`（`Models.cs:174-239`）加两个类：ctor `base(MDModelKind.MD_MODEL_VEHICLE_KEYPOINT, modelPath, opt)` / `base(MDModelKind.MD_MODEL_FACE_LANDMARK, ...)`；`Clone()`；`Predict` 复用 `ReadHand`/`ReadPose` 私有静态读取器逻辑（`md_result_pose` + `md_result_keypoints` → `PoseResult[]`）；`SetKeypointsNum`/`SetConfThreshold`/`SetNmsThreshold` 仅 Vehicle 暴露；`PredictBatch` 复用 `ReadHandBatch`。可将 `HandModel` 的读取器改名为共享私有静态（如 `ReadKeypointsPose`）供三者复用（DRY）——若改动面大，也可在 `HandModel` 保留原样、新模型复制一份读取器。
- [ ] **Step 3**: `AllModelsTests.cs` 仿 `Hand_Works`（`AllModelsTests.cs:119`）加：
```csharp
[Test]
public void VehicleKeypoint_Works() {
    var model = Path.Combine(ModelRoot, "vehicle_keypoint.onnx");
    var img = Path.Combine(ImageRoot, "bus.jpg");
    if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
    using var vi = VisionImage.Read(img);
    using var m = new VehicleKeypointModel(model, CpuOrt());
    m.SetKeypointsNum(4);
    var r = m.Predict(vi);
    Assert.That(r, Is.Not.Empty);
    Assert.That(r[0].KeyPoints.Length, Is.GreaterThan(0));
}

[Test]
public void FaceLandmark_Works() {
    var model = Path.Combine(ModelRoot, "2d106det.onnx");
    var img = Path.Combine(ImageRoot, "face.jpg");
    if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
    using var vi = VisionImage.Read(img);
    using var m = new FaceLandmarkModel(model, CpuOrt());
    var r = m.Predict(vi);
    Assert.That(r, Is.Not.Empty);
    Assert.That(r[0].KeyPoints.Length, Is.EqualTo(106));
}
```
- [ ] **Step 4**: `dotnet build` + `dotnet test --filter "VehicleKeypoint_Works|FaceLandmark_Works"`。
- [ ] **Step 5: Commit**

```bash
git add csharp/ModelDeploy/*.cs csharp/ModelDeployUnitTest/AllModelsTests.cs
git commit -m "feat(csharp): VehicleKeypointModel + FaceLandmarkModel wrappers"
```

---

### Task 6: Rust（`VehicleKeypoint` / `FaceLandmark`）

> **降级判断（推荐保留）**：Rust 有 `HandKeypoint` 的 `model_wrapper!` 完整模板（`model.rs:1303`，`RawResult::pose/pose_batch` → `Vec<Pose>`），新增仅"复制改名 + ModelKind + 枚举"，成本极低。**建议保留**；若时间紧可 DEV-SKIPPED（C++/pybind/CAPI 已交付）。

**Files:**
- Modify: `rust/modeldeploy/src/ffi.rs`（`MDModelKind` 枚举加 `VEHICLE_KEYPOINT`/`FACE_LANDMARK`）
- Modify: `rust/modeldeploy/src/types.rs`（`ModelKind` 加两变体 + `to_ffi` match）
- Modify: `rust/modeldeploy/src/model.rs`（`ResultType` impl + `model_wrapper!`）
- Modify: `rust/modeldeploy/src/lib.rs`（重导出）
- Test: `rust/modeldeploy/tests/integration_test.rs`（`test_vehicle_keypoint` / `test_face_landmark`）

**Interfaces:**
- Consumes: CAPI (Task 4)
- Produces: Rust `modeldeploy::VehicleKeypoint` / `FaceLandmark`，`new(path, &opt)`，`predict(&Image) -> Result<Vec<Pose>>`。

- [ ] **Step 1**: `ffi.rs` `MDModelKind` 枚举加 `VEHICLE_KEYPOINT`/`FACE_LANDMARK`（值对齐 CAPI 顺序）。`types.rs` `ModelKind` 加 `VehicleKeypoint`/`FaceLandmark`，`to_ffi` match 加 `ModelKind::VehicleKeypoint => VEHICLE_KEYPOINT,` / `ModelKind::FaceLandmark => FACE_LANDMARK,`。
- [ ] **Step 2**: `model.rs`：加 `impl ResultType for VehicleKeypoint { type Item = Pose; }` / `impl ResultType for FaceLandmark { type Item = Pose; }`；加 `model_wrapper!(VehicleKeypoint, ModelKind::VehicleKeypoint, RawResult::pose, RawResult::pose_batch);` / `model_wrapper!(FaceLandmark, ModelKind::FaceLandmark, RawResult::pose, RawResult::pose_batch);`（镜像 `HandKeypoint`，`model.rs:1303`）。`lib.rs` 重导出 `VehicleKeypoint`/`FaceLandmark`。
- [ ] **Step 3**: `integration_test.rs` 仿 `test_hand_keypoint`（`integration_test.rs:179`）加：
```rust
#[test]
fn test_vehicle_keypoint() -> Result<(), MdError> {
    let opt = RuntimeOption::default();
    let path = test_data("test_models/onnx/vehicle_keypoint.onnx");
    if !path.exists() { eprintln!("vehicle_keypoint.onnx missing (external weights), skipping"); return Ok(()); }
    let model = VehicleKeypoint::new(&path, &opt)?;
    let img = Image::new(&test_data("images/bus.jpg"))?;
    let r = model.predict(&img)?;
    assert!(!r.is_empty());
    assert!(r[0].keypoints.len() > 0);
    Ok(())
}

#[test]
fn test_face_landmark() -> Result<(), MdError> {
    let opt = RuntimeOption::default();
    let path = test_data("test_models/onnx/2d106det.onnx");
    if !path.exists() { eprintln!("2d106det.onnx missing (external weights), skipping"); return Ok(()); }
    let model = FaceLandmark::new(&path, &opt)?;
    let img = Image::new(&test_data("images/face.jpg"))?;
    let r = model.predict(&img)?;
    assert!(!r.is_empty());
    assert_eq!(r[0].keypoints.len(), 106);
    Ok(())
}
```
（若 `Pose.keypoints` 字段名与既有模板不同，以 `types.rs:55` `Pose` 结构为准校正；需顶部 `use modeldeploy::{FaceLandmark, VehicleKeypoint, ...}`。）
- [ ] **Step 4**: `cargo build` + `cargo test test_vehicle_keypoint test_face_landmark`；`cargo clippy` 干净。
- [ ] **Step 5: Commit**

```bash
git add rust/modeldeploy/src/*.rs rust/modeldeploy/tests/integration_test.rs
git commit -m "feat(rust): VehicleKeypoint + FaceLandmark bindings"
```

---

### Task 7: `demo_landmark`（一次跑两个模型）+ docs

**Files:**
- Create: `examples/demo_landmark/demo_landmark.cpp`
- Create: `examples/demo_landmark/CMakeLists.txt`
- Modify: `examples/CMakeLists.txt`（`add_subdirectory(demo_landmark)`，在 `demo_action` 之后）
- Modify: `examples/EXAMPLES.md`（加行）、`README.md`（能力加"面部 Landmark / 车辆关键点"）

**Interfaces:**
- Consumes: `landmark::VehicleKeypoint`/`FaceLandmark` (Task 1/2)

- [ ] **Step 1: 写 `examples/demo_landmark/CMakeLists.txt`**（镜像 `demo_hand/CMakeLists.txt`）

```cmake
add_executable(demo_landmark demo_landmark.cpp)
target_link_libraries(demo_landmark PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
```

- [ ] **Step 2: 写 `examples/demo_landmark/demo_landmark.cpp`**

```cpp
// ModelDeploy demo_landmark：关键点扩展（车辆关键点 + 面部 Landmark 106 点）。
// Usage: demo_landmark <vehicle.onnx> <face.onnx> <image.jpg>
//   写第一个参数为 "none" 表示跳过车辆模型（无需对应权重）。
//   写第二个参数为 "none" 表示跳过人脸模型。
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/landmark/face_landmark.h"
#include "vision/landmark/vehicle_keypoint.h"

static void run_vehicle(const std::string& modelFile, const std::string& imgFile) {
    if (modelFile == "none") return;
    modeldeploy::RuntimeOption opt; opt.use_ort_backend();
    modeldeploy::vision::landmark::VehicleKeypoint model(modelFile, opt);
    if (!model.is_initialized()) { printf("vehicle init failed (missing weights?)\n"); return; }
    auto im = cv::imread(imgFile);
    if (im.empty()) { printf("cannot read image %s\n", imgFile.c_str()); return; }
    modeldeploy::vision::ImageData img(im);
    std::vector<modeldeploy::vision::KeyPointsResult> res;
    if (!model.predict(img, &res)) { printf("vehicle predict failed\n"); return; }
    printf("vehicle: %zu object(s), keypoints_num=%zu\n", res.size(),
           res.empty() ? 0 : res[0].keypoints.size());
    for (auto& r : res)
        printf("  box=(%.1f,%.1f,%.1f,%.1f) score=%.3f keypoints=%zu\n",
               r.box.x, r.box.y, r.box.width, r.box.height, r.score, r.keypoints.size());
}

static void run_face(const std::string& modelFile, const std::string& imgFile) {
    if (modelFile == "none") return;
    modeldeploy::RuntimeOption opt; opt.use_ort_backend();
    modeldeploy::vision::landmark::FaceLandmark model(modelFile, opt);
    if (!model.is_initialized()) { printf("face init failed (missing weights?)\n"); return; }
    auto im = cv::imread(imgFile);
    if (im.empty()) { printf("cannot read image %s\n", imgFile.c_str()); return; }
    modeldeploy::vision::ImageData img(im);
    std::vector<modeldeploy::vision::KeyPointsResult> res;
    if (!model.predict(img, &res)) { printf("face predict failed\n"); return; }
    printf("face: %zu result(s), keypoints_num=%zu\n", res.size(),
           res.empty() ? 0 : res[0].keypoints.size());
    for (auto& r : res)
        printf("  box=(%.1f,%.1f,%.1f,%.1f) keypoints=%zu\n",
               r.box.x, r.box.y, r.box.width, r.box.height, r.keypoints.size());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: demo_landmark <vehicle.onnx|none> <face.onnx|none> <image.jpg>\n");
        return 1;
    }
    run_vehicle(argv[1], argv[3]);
    run_face(argv[2], argv[3]);
    return 0;
}
```

- [ ] **Step 3: `examples/CMakeLists.txt` + EXAMPLES.md + README.md**

`examples/CMakeLists.txt` 加 `add_subdirectory(demo_landmark)`（`demo_action` 之后）。
`EXAMPLES.md` 加行：
```
| `demo_landmark` | 关键点扩展（车辆关键点 / 面部 Landmark 106 点） | `onnx/vehicle_keypoint/*.onnx|onnx/2d106det/*.onnx` | 图片 jpg | 打印各类键点数量与坐标 |
```
`README.md` 能力列表"……面部 Landmark / 车辆关键点"加到关键点相关能力处（如"姿态/手部关键点"附近）。

- [ ] **Step 4: 构建 + 运行**

Run（`BUILD_VISION=ON` 构建）：`cmake --build build --parallel 8`；`.\bin\demo_landmark.exe`（无参 Usage）。
Expected: 编译 0 errors；Usage 正常；`none` 跳过路径与缺权重路径清晰报错不崩溃。

- [ ] **Step 5: Commit**

```bash
git add examples/demo_landmark/ examples/CMakeLists.txt examples/EXAMPLES.md README.md
git commit -m "feat(examples): demo_landmark (VehicleKeypoint + FaceLandmark) + docs"
```

---

### Task 8: 全量验证 + 跨后端语义确认 + 真实权重联调

**Files:** 无新增（验证；必要时 minor 修复）

**Interfaces:**
- Consumes: 全部前序任务

- [ ] **Step 1: 全量 C++ 测试** — `.\bin\test_modeldeploy.exe "[landmark]"`、`"[capi]"`、`"[core]"`、`"[vision_models]"`。记录通过数；无回归。
- [ ] **Step 2: 跨后端语义确认** — grep 确认 `vehicle_keypoint.cpp`/`face_landmark.cpp` 仅依赖 `UltralyticsPose`/`InsightFaceLandmark`（二者均自身只吃 `BaseModel`/`get_input_info`），无 ORT/MNN/TRT 直接 include → 语义一致。
- [ ] **Step 3: 绑定量测** — Python（`landmark.VehicleKeypoint/FaceLandmark` 冒烟）、C#（`VehicleKeypoint_Works`/`FaceLandmark_Works`）、Rust（`test_vehicle_keypoint`/`test_face_landmark`）、`demo_landmark`。
- [ ] **Step 4: 真实权重联调校正** — 若拿到 `vehicle_keypoint.onnx`/`2d106det.onnx` 权重：校正 `FaceLandmark` 的 bbox/尺寸假设（裁剪图 vs 需 insightface 192×192 对齐）、`vis_keypoints` 实参细节、Vehicle `set_keypoints_num` 与真实输出点数；更新相关注释。
- [ ] **Step 5: 收尾报送** — 报告 + concerns（如 FaceLandmark 走专用实现 vs spec 初案 pose 复用的偏差、Vehicle 默认 4 点选择、insightface 尺寸/bbox 假设）。

---

## Self-Review 记录

**Spec coverage**：
- 架构目录/命名空间(§3.1) → Task 1/2 建 `csrc/vision/landmark/`，`landmark` 命名空间。
- `FaceLandmark`(§3.2) → Task 2（**关键偏差**：spec §3.2 初案"复用 pose 泛化"，因 insightface 106 模型与 pose 管线不兼容改走 §12 专用实现出口——内部复用 `InsightFaceLandmark`，`predict(裁剪图)` 适配 106 点到 `KeyPointsResult`）。
- `VehicleKeypoint`(§3.3) → Task 1（薄封装 `UltralyticsPose` + `set_keypoints_num(4)`，输出 `KeyPointsResult`）。
- Python(§4) → Task 3；CAPI(§5) → Task 4（复用 `MD_RES_POSE`+`md_result_keypoints`，与 HAND 一致）；C#(§6)/Rust(§7) → Task 5/6（薄封装，标注可 DEV-SKIPPED）。
- demo+docs(§8) → Task 7（`demo_landmark` 一次跑两模型）；测试(§9) → Task 1/2 的 `[landmark]` 单测 + 缺权重 SKIP；Task 4 `[capi]`；Task 8 验证。
- 交付矩阵(§10) → C++/pybind/CAPI 全量，C#/Rust 薄封装，demo/docs 全量。
- 已知限制/风险(§11/12) → 在 Global Constraints 与 Task 2 决策块显式记录：FaceLandmark 走出口、权重外链 SKIP、Vehicle 点数泛化。

**Placeholder 扫描**：所有代码步骤均含完整实现或"以现有 HAND 分支为准"的明确复制指令，无 "TBD/TODO" 占位。`vis_keypoints` 实参细节、`InsightFaceLandmark` 尺寸假设为**集成时校正**的数值/实参点（Task 1/2/8 已注明模板与校正位置），非未定义接口。

**Type consistency**：
- `VehicleKeypoint::predict(const ImageData&, vector<KeyPointsResult>*, TimerArray*)` / `FaceLandmark::predict(...)` 在 Task 1/2/3/4/7 一致。
- `UltralyticsPosePostprocessor` 的 `set/get_keypoints_num` 在 Task 1/3/4/5/7 用法一致；`VehicleKeypoint` 默认 `keypoints_num=4` 在 Task 1（构造）、Task 3（pybind get==4）一致。
- 枚举名 `MD_MODEL_VEHICLE_KEYPOINT`/`MD_MODEL_FACE_LANDMARK` 在 Task 4/5/6 全链一致；Rust `ModelKind::VehicleKeypoint/FaceLandmark` 与 CAPI 顺序对齐。
- 结果复用 `KeyPointsResult` + `MD_RES_POSE` + `md_result_keypoints` 在 Task 2/4/5/6 一致；C# `PoseResult`/Rust `Pose` 复用既有读取器。

---

## 参考实现（切入源码）

- 前端模板（逐字对照）：`csrc/vision/hand/hand.{h,cpp}`
- 泛化后端：`csrc/vision/pose/ultralytics_pose.{h,cpp}`、`csrc/vision/pose/postprocessor.h:46`
- 面部专用后端（复用）：`csrc/vision/face/insightface/landmark/insightface_landmark.{h,cpp}`
- 结果结构：`csrc/vision/common/result.h:80`；可视化：`csrc/vision/common/visualize/visualize.h:56/62`
- pybind 模板：`csrc/pybind/vision/hand_pybind.cpp`；注册：`csrc/pybind/vision/vision_pybind.cpp`
- CAPI 模板（MD_MODEL_HAND 全链）：`capi/md_capi.cpp:1013/1074/1132/1244/1277/1367/1592/1881`
- C# 模板：`csharp/ModelDeploy/Models.cs:174`；枚举：`types_internal_c.cs:62`；测试：`AllModelsTests.cs:119`
- Rust 模板：`rust/modeldeploy/src/model.rs:1303`；枚举：`types.rs:173`；测试：`integration_test.rs:179`
- demo 模板：`examples/demo_hand/`；examples 注册：`examples/CMakeLists.txt`
- 测试模板：`tests/test_hand.cpp`、`tests/CMakeLists.txt`

## 已知风险 / 开放问题

- **FaceLandmark 专用实现 vs pose 复用**（已定）：insightface 106 无法复用 pose 后端，走 §12 出口；若未来出现 YOLO-pose 格式人脸关键点模型，可另加 pose 后端分支（本计划默认专用实现）。
- **`FaceLandmark::predict` 的 bbox 假设**：以整裁剪图为 bbox 估计 106 点。对"整图 + 单脸居中"合理；若用户传入含多脸整图，需先人脸裁剪（YAGNI，不在此 Item 范围）。
- **insightface 尺寸**：`InsightFaceLandmark` 内部按 `input_size_{192,192}` 处理，真实权重联调(Task 8)校验裁剪图输入尺寸是否需上游 resize 到 192×192。
- **Vehicle 默认 4 点**：spec 说 4-8 点；以 4（车轮）为默认，模型不同用 `set_keypoints_num` 覆盖。
- **C#/Rust 是否 DEV-SKIPPED**：推荐保留（模板现成、极低成本）；若时间紧，FaceLandmark(或两者) 标注 DEV-SKIPPED，C++/pybind/CAPI 已覆盖核心能力。
