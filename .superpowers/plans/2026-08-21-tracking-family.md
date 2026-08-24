# 追踪家族（ByteTrack / BoT-SORT / StrongSORT）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用纯 C++ 实现三种多目标追踪器（ByteTrack、BoT-SORT、StrongSORT），统一 Tracker 接口，ReID 外观经 ORT 推理，并绑定 C API / C# / Python / Rust。

**Architecture:** 新建 `csrc/vision/tracking/` 组件化目录（base_tracker 统一接口 + bytetrack/botsort/strongsort + reid_extractor + matching 纯几何组件 + mot 测试工具）。输入每帧检测框 → 输出带稳定 track_id 的 TrackResult。视觉源码由根 CMake GLOB 自动收集，无需改 CMakeLists。

**Tech Stack:** C++17、现有 modeldeploy::Runtime（ORT）、OpenCV（裁剪/ECC/可视化）、Catch2 测试、pybind11 / C API / C# / Rust。

## Global Constraints

- C++17（根 CMakeLists 已设置）
- MSVC 需 `/utf-8`（根 CMakeLists 已为 SDK 自动设置）
- vision 源码通过根 CMake `file(GLOB_RECURSE VISION_SOURCE ...)` 收集，**不要新增子目录 CMakeLists**
- 复用现有导出宏 `MODELDEPLOY_CXX_EXPORT`（`core/md_decl.h`）
- ReID ONNX 权重与 MOT 测试数据走 modelscope 下载至 test_data，**不进仓库**
- 测试数据当前不在仓库（AGENTS.md 说明需单独下载 test_data.zip）
- 数据结构复用 `csrc/vision/common/result.h`（DetectionResult/Rect2f）、`csrc/vision/common/struct.h`（ImageData）

---

### Task 1: 公共数据结构 + 统一 Tracker 接口（TDD）

**Files:**
- Create: `csrc/vision/tracking/base_tracker.h`
- Create: `csrc/vision/tracking/base_tracker.cpp`
- Create: `tests/test_tracking.cpp`
- Modify: `tests/CMakeLists.txt`（把 `test_tracking.cpp` 加进 `TEST_SOURCES` 列表——注意 tests 用**显式列表**而非 GLOB）

**Interfaces:**
- Produces:
  - `struct Detection { Rect2f box; float score; int label_id; std::vector<float> feature; }`
  - `struct TrackResult { int track_id; Rect2f box; float score; int label_id; int state; std::vector<float> feature; }`
  - `enum class TrackState : int { New=0, Tracked=1, Lost=2, Removed=3 };`
  - `class BaseTracker { virtual std::vector<TrackResult> update(const std::vector<Detection>& detections, const ImageData* frame=nullptr, double timestamp=-1)=0; virtual void reset()=0; virtual ~BaseTracker()=default; };`
  - `std::vector<TrackResult> empty_update()` helper（原子测试用）

- [ ] **Step 1: 写失败测试**

```cpp
// tests/test_tracking.cpp
#include <catch2/catch_test_macros.hpp>
#include "vision/tracking/base_tracker.h"
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::tracking;

TEST_CASE("BaseTracker: empty input yields empty output", "[tracking]") {
    auto r = empty_update();
    REQUIRE(r.empty());
}
```

- [ ] **Step 2: 运行确认失败（编译失败）**

Run: `cmake --build build_tdc_gpu --target test_modeldeploy && cd build_tdc_gpu && ctest -R tracking`
Expected: 编译失败，"vision/tracking/base_tracker.h no such file"（本机验证环境为 `build_tdc_gpu`，即 WITH_GPU=ON + BUILD_TESTS=ON 的 ninja 增量构建；`build` 目录是 WSL 产物不可用）

- [ ] **Step 3: 写最小实现**

`csrc/vision/tracking/base_tracker.h`：
```cpp
#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision::tracking {
    struct MODELDEPLOY_CXX_EXPORT Detection {
        Rect2f box; float score{}; int label_id{};
        std::vector<float> feature;
    };
    struct MODELDEPLOY_CXX_EXPORT TrackResult {
        int track_id{-1}; Rect2f box; float score{}; int label_id{};
        int state{0}; std::vector<float> feature;
    };
    enum class TrackState : int { New = 0, Tracked = 1, Lost = 2, Removed = 3 };
    class MODELDEPLOY_CXX_EXPORT BaseTracker {
    public:
        virtual std::vector<TrackResult> update(
            const std::vector<Detection>& detections,
            const ImageData* frame = nullptr, double timestamp = -1) = 0;
        virtual void reset() = 0;
        virtual ~BaseTracker() = default;
    };
    inline std::vector<TrackResult> empty_update() { return {}; }
}
```

- [ ] **Step 4: 运行测试确认通过**
Run: `cmake --build build --target test_modeldeploy && cd build && ctest -R tracking`
Expected: PASS

- [ ] **Step 5: 提交**
```bash
git add csrc/vision/tracking/base_tracker.h tests/test_tracking.cpp
git commit -m "feat(track): BaseTracker interface + Detection/TrackResult structs"
```

---

### Task 2: 卡尔曼滤波 + 匈牙利匹配 + IoU 代价（纯几何组件）

**Files:**
- Create: `csrc/vision/tracking/matching/iou_matching.h/.cpp`
- Create: `csrc/vision/tracking/matching/hungarian.h/.cpp`
- Create: `csrc/vision/tracking/matching/kalman_filter.h/.cpp`
- Test: `tests/test_tracking.cpp`

**Interfaces:**
- Produces:
  - `float iou(const Rect2f& a, const Rect2f& b);`
  - `std::vector<std::vector<float>> iou_distance(const std::vector<Rect2f>& a, const std::vector<Rect2f>& b);`
  - `std::vector<std::pair<int,int>> linear_sum_assignment(const std::vector<std::vector<float>>& cost);`
  - `class KalmanFilter { void init(const Rect2f&); void predict(); void update(const Rect2f&); Rect2f get_state() const; };`

- [ ] **Step 1: 写失败测试**
```cpp
TEST_CASE("Matching: IoU overlap", "[tracking]") {
    Rect2f a{0,0,10,10}, b{0,0,10,10};
    REQUIRE(iou(a,b) == Approx(1.0f));
}
TEST_CASE("Matching: hungarian assignment", "[tracking]") {
    std::vector<std::vector<float>> cost{{1,2},{2,1}};
    auto res = linear_sum_assignment(cost);
    REQUIRE(res.size() == 2);
}
```
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**（IoU 交并比；匈牙利用经典 O(n³) 算法；卡尔曼为 7 维 x,y,a(=w/h),h,vx,va,vh 线性匀速，SORT 官方公式）
- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**
```bash
git add csrc/vision/tracking/matching/ tests/test_tracking.cpp
git commit -m "feat(track): kalman filter + hungarian + iou matching"
```

---

### Task 3: ByteTrack 追踪器

**Files:**
- Create: `csrc/vision/tracking/bytetrack.h/.cpp`
- Test: `tests/test_tracking.cpp`

**Interfaces:**
- Consumes: `BaseTracker`, `KalmanFilter`, `linear_sum_assignment`, `iou_distance` (Tasks 1-2)
- Produces: `class ByteTracker : public BaseTracker { void set_params(float track_thresh=0.5, float high_thresh=0.5, float low_thresh=0.1, int max_age=30, int min_hits=3, float iou_threshold=0.3); };`

**ByteTrack 算法（SOTA 对齐）：**
- 每帧：高置信检测框（score>high_thresh）与 track 做 IoU 匹配 → 更新
- 剩余 track 与低置信框（low_thresh<score<high_thresh）再匹配 → 抗遮挡
- 未匹配的 box 初始化新 track；未匹配的 track 计数 lost_at；超过 max_age 移除
- 卡尔曼 predict 更新中心；输出 state=Tracked/Lost/New

- [ ] **Step 1: 写失败测试（合成轨迹：2 帧同一个 box → 同 track_id）**
```cpp
TEST_CASE("ByteTrack: stable id across frames", "[tracking]") {
    ByteTracker tr;
    Detection d1{{0,0,20,20},0.9f,0};
    auto f1 = tr.update({d1});
    Detection d2{{2,2,20,20},0.9f,0};
    auto f2 = tr.update({d2});
    REQUIRE(f1.size()==1); REQUIRE(f2.size()==1);
    REQUIRE(f1[0].track_id == f2[0].track_id);
}
TEST_CASE("ByteTrack: lost keeps id within max_age", "[tracking]") {
    ByteTracker tr;
    Detection d{{0,0,20,20},0.9f,0};
    tr.update({d});                          // 检出
    auto miss = tr.update({});               // 丢失一帧
    auto back = tr.update({d});              // 回到视野
    REQUIRE(back.size()==1);
    REQUIRE(back[0].track_id == 0);          // id 复用
}
```
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现 ByteTracker::update**（如上算法，用 Boyer's 官方 ByteTrack 流程）
- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**
```bash
git add csrc/vision/tracking/bytetrack.h/.cpp tests/test_tracking.cpp
git commit -m "feat(track): ByteTrack tracker"
```

---

### Task 4: ReID 外观提取器（reid_extractor）

**Files:**
- Create: `csrc/vision/tracking/reid_extractor.h/.cpp`
- Test: `tests/test_tracking.cpp`

**Interfaces:**
- Consumes: `BaseModel` / `Runtime`, `ImageData`, `Tensor`
- Produces: `class ReidExtractor { bool init(const std::string& onnx, const RuntimeOption& opt); std::vector<float> extract(const ImageData& patch); };`

- [ ] **Step 1: 写失败测试**（用模型文件存在性 + 空输入返回空）
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**（加载 OSNet ONNX；预处理 resize e.g.256×128 + normalize；infer → 输出 L2 归一化特征；用 BaseModel 机制）
- [ ] **Step 4: 运行确认通过**（测试数据若无 onnx 则跳过，同 test_vision_models 的 `if(!fs::exists) return;` 模式）
- [ ] **Step 5: 提交**

---

### Task 5: BoT-SORT 追踪器（ByteTrack + ReID + CMC）

**Files:**
- Create: `csrc/vision/tracking/botsort.h/.cpp`
- Test: `tests/test_tracking.cpp`

**Interfaces:**
- Consumes: `BaseTracker`, `KalmanFilter`, `ReidExtractor`, `iou_distance` (Tasks 1-4)
- Produces: `class BotSortTracker : public BaseTracker { void set_params(...); void set_reid(std::shared_ptr<ReidExtractor>); };`

**BoT-SORT 算法（SOTA 对齐）：**
- 基于 ByteTrack 两阶段，但匹配用 IoU+外观余弦距离融合代价
- EMA 维护每个 track 的特征表征
- CMC（Camera Motion Compensation）：两帧间 ECC（cv::findTransformECC）估计仿射/单应，校正预测框
- Interpolate 可选

- [ ] **Step 1: 写失败测试**（合成轨迹：不同外观同位置 → 不同 id；EMA 特征平均）
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

---

### Task 6: StrongSORT 追踪器（ReID + NSA Kalman + EMA）

**Files:**
- Create: `csrc/vision/tracking/strongsort.h/.cpp`
- Test: `tests/test_tracking.cpp`

**Interfaces:**
- Consumes: `BaseTracker`, `ReidExtractor`, kalman (NSA), `iou_distance` (Tasks 1-4)
- Produces: `class StrongSortTracker : public BaseTracker { void set_params(...); void set_reid(std::shared_ptr<ReidExtractor>); };`

**StrongSORT 算法（SOTA 对齐）：**
- NSA Kalman（噪声尺度自适应）
- 深度特征关联：EMA 特征库 + 余弦距离（外观优先）
- ECC 全局运动校正 + 可选插值

- [ ] **Step 1: 写失败测试**
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

---

### Task 7: MOT 测试 harness（加载 MOT 序列 + CLEAR 指标）

**Files:**
- Create: `csrc/vision/tracking/mot/mot_loader.h/.cpp`
- Create: `csrc/vision/tracking/mot/clear_metrics.h/.cpp`
- Test: `tests/test_tracking.cpp`

**Interfaces:**
- Produces: `struct MotFrame { int frame_id; std::vector<Detection> dets; }; std::vector<MotFrame> load_mot_dets(const fs::path&);`
- `struct Metrics { float mota, idf1, hota, num_switches; }; Metrics compute_clear(const std::vector<std::vector<TrackResult>>&);`

- [ ] **Step 1: 写失败测试**（空序列 → 0/1）
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**（MOTChallenge GT 格式解析；MOTA=1-(FP+FN+IDS)/GT；IDF1 基于 ID 匹配；HOTA 简化版）
- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

---

### Task 8: 合成 MOT 序列端到端测试（三种追踪器对比）

**Files:**
- Modify: `tests/test_tracking.cpp`

- [ ] **Step 1: 写失败测试**（合成 2 目标 30 帧，ByteTrack 应保持 2 条稳定 track，MOTA 接近 1）
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现测试数据生成 + 断言**
- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

---

### Task 9: Python 绑定

**Files:**
- Create: `csrc/pybind/vision/tracking_pybind.cpp`
- Modify: `csrc/pybind/vision/vision_pybind.cpp`（声明 + 调用 `bind_tracking`）

- [ ] **Step 1: 写绑定代码**
```cpp
#include <pybind11/pybind11.h>
#include "vision/tracking/bytetrack.h"
#include "vision/tracking/botsort.h"
#include "vision/tracking/strongsort.h"
namespace modeldeploy::vision {
void bind_tracking(const pybind11::module& m) {
    using namespace tracking;
    pybind11::class_<Detection>(m,"Detection")
        .def(pybind11::init<>()).def_readwrite("box",&Detection::box)
        .def_readwrite("score",&Detection::score).def_readwrite("label_id",&Detection::label_id);
    pybind11::class_<TrackResult>(m,"TrackResult")
        .def_readonly("track_id",&TrackResult::track_id).def_readonly("box",&TrackResult::box)
        .def_readonly("state",&TrackResult::state);
    pybind11::class_<ByteTracker>(m,"ByteTracker")
        .def(pybind11::init<>())
        .def("update", &ByteTracker::update)
        .def("reset", &ByteTracker::reset);
}
}
```
- [ ] **Step 2: vision_pybind.cpp 加声明 `void bind_tracking(const pybind11::module&);` 并在 `bind_vision` 调用**
- [ ] **Step 3: 构建 `cmake --build build --target modeldeploy`（python 模块）**
- [ ] **Step 4: 验证 `python -c "from modeldeploy import vision; ..."`**
- [ ] **Step 5: 提交**

---

### Task 10: C API 绑定

**Files:**
- Modify: `capi/md_capi.h`、`capi/md_capi.cpp`
- Modify: `tests/test_capi.cpp`

- [ ] **Step 1: 加句柄与函数声明**
```c
typedef struct md_tracker* MDTrackerHandle;
MDStatus md_tracker_create(int algo, MDTrackerHandle* out); // algo: 0=Byte,1=BoT,2=Strong
MDStatus md_tracker_update(MDTrackerHandle, const MDBox* boxes, size_t n, MDTrackResult* out, size_t* count);
MDStatus md_tracker_release(MDTrackerHandle);
```
（MDBox/MDTrackResult 复用现有 box/result 的 C 结构，若不存在则新增）
- [ ] **Step 2: 实现**
- [ ] **Step 3: 构建 + 测试 test_capi**
- [ ] **Step 4: 提交**

---

### Task 11: C# 绑定

**Files:**
- Modify: `csharp/ModelDeploy/Tracker.cs`（新建）、`csharp/ModelDeploy/NativeMethods.cs`

- [ ] **Step 1: NativeMethods.cs 加 DllImport（md_tracker_create/update/release）**
- [ ] **Step 2: Tracker.cs 封装 ByteTracker/BotSortTracker/StrongSortTracker + TrackResult**
- [ ] **Step 3: `dotnet build` 验证**
- [ ] **Step 4: 提交**

---

### Task 12: Rust 绑定

**Files:**
- Create: `rust/modeldeploy/src/tracker.rs`
- Modify: `rust/modeldeploy/src/lib.rs`

- [ ] **Step 1: tracker.rs 暴露 ByteTracker/BotSortTracker/StrongSortTracker + TrackResult**
- [ ] **Step 2: lib.rs `pub mod tracker;`**
- [ ] **Step 3: `cargo build` 验证**
- [ ] **Step 4: 提交**

---

### Task 13: 端到端 demo + 文档

**Files:**
- Create: `examples/demo_tracking/demo_tracking.cpp`（det→track→可视化）
- Modify: `examples/EXAMPLES.md`

- [ ] **Step 1: demo 实现**（UltralyticsDet.predict 每帧 → ByteTracker.update → 画框+id）
- [ ] **Step 2: EXAMPLES.md 补充**
- [ ] **Step 3: 全量构建 + 测试 `cmake --build build && ctest`**
- [ ] **Step 4: 提交**

---

## Self-Review 记录

- **Spec 覆盖**：统一接口(T1)、ByteTrack(T3)、BoT-SORT(T5)、StrongSORT(T6)、ReID(T4)、绑定 Python(T9)/CAPI(T10)/C#(T11)/Rust(T12)、MOT 测试(T7/T8)、demo(T13)。✓
- **占位符扫描**：无 TBD/TODO；Task 4-6 的实现细节标注了算法要点与接口，具体逻辑在实现时按官方 SOTA 公式落地。✓
- **类型一致性**：Detection/TrackResult/BaseTracker::update 三处（T1 定义、T9 Python、T10 CAPI）签名一致。✓
