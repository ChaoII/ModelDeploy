# OBB NMS 优化 + with-NMS(end2end) 验证 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 优化 SDK 的 `utils::obb_nms`（AABB 预筛 + 提拉 cv 转换，结果逐位不变），并用 ORT-CPU 验证带 NMS 的 `yolo26n-obb-end2end` 模型走 `run_with_nms` 路径可用（post≈0）。

**Architecture:** 优化集中在 `csrc/vision/obb_nms.cpp` 的 `obb_nms` 内部（不改任何接口/分发）；带 NMS 模型通过在 `benchmark/benchmark_models.cpp` 新增独立用例（显式 `set_size({1024,1024})`、仅 ORT-CPU）验证。新增一个 `tests/test_obb_nms.cpp` 单元测试，用「旧算法参考实现」断言新旧 `obb_nms` 输出逐位一致，作为重构的回归护栏。

**Tech Stack:** C++17, OpenCV(`cv::RotatedRect`/`cv::Rect2f`/`cv::rotatedRectangleIntersection`), Catch2, ONNX Runtime (ORT-CPU), Ninja+MSVC。

## Global Constraints

- 必须保留 `utils::obb_nms(std::vector<ObbResult>*, float, std::vector<int>*)` 签名不变（`csrc/vision/utils.h:47`）。
- AABB 早筛是**等价变换**：旋转矩形相交 ⇒ 外接矩形必相交；外接矩形不相交 ⇒ 旋转 IoU=0 ≤ 阈值，不改变抑制决策。禁止改动排序规则与抑制规则。
- 不改变 postprocessor/preprocessor 分发逻辑；不改对外 API。
- MSVC 需 `/utf-8`（根 CMakeLists 已为 SDK 自动设置）；`CMAKE_CXX_STANDARD=17`；构建使用 Ninja：`ninja -C build benchmark` / `ninja -C build test_modeldeploy`。
- 仅在 `tests/CMakeLists.txt` 的 `TEST_SOURCES` 列表显式加入新测试文件（该列表是显式的，非 GLOB）。
- 现有基线/测试（如 `test_vision_models.cpp:398` `[vision_models]` OBB 用例）必须保持通过。

---

### Task 1: 优化 `obb_nms` + 新增等价性单元测试

**Files:**
- Modify: `csrc/vision/obb_nms.cpp:118-158`（`obb_nms` 函数体）
- Create: `tests/test_obb_nms.cpp`
- Modify: `tests/CMakeLists.txt`（`TEST_SOURCES` 加 `test_obb_nms.cpp`）
- Test: `build/bin/test_modeldeploy [core]`

**Interfaces:**
- Consumes: `utils::rotated_rect_to_cv_type(const RotatedRect) -> cv::RotatedRect`（`utils.cpp:174` / `utils.h:38`）、本文件 `float rotated_iou(const cv::RotatedRect&, const cv::RotatedRect&)`（`obb_nms.cpp:89`）。
- Produces: 保持 `utils::obb_nms` 签名与语义不变；`tests/test_obb_nms.cpp` 中的 `reference_obb_nms` 为测试内部函数。

- [ ] **Step 1: 创建单元测试 `tests/test_obb_nms.cpp`**

```cpp
//
// Created by the ModelDeploy team on 2026/8/20.
//
#include <vector>
#include <numeric>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <catch2/catch_test_macros.hpp>
#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/common/struct.h"
#include "vision/common/result.h"

using modeldeploy::vision::RotatedRect;
using modeldeploy::vision::ObbResult;
using modeldeploy::vision::utils::obb_nms;
using modeldeploy::vision::utils::rotated_rect_to_cv_type;

namespace {
    // 与 obb_nms.cpp 中 rotated_iou(cv, cv) 完全相同的 IoU 计算（cv 多边形相交）
    float ref_rotated_iou(const cv::RotatedRect& a, const cv::RotatedRect& b) {
        std::vector<cv::Point2f> inter;
        if (cv::rotatedRectangleIntersection(a, b, inter) <= 0) return 0.0f;
        const float inter_area = static_cast<float>(cv::contourArea(inter));
        const float union_area = a.size.area() + b.size.area() - inter_area;
        return union_area > 0 ? inter_area / union_area : 0.0f;
    }

    // 旧实现（O(N^2) 朴素）作为参考金标准 —— 新实现必须与之逐位一致
    void reference_obb_nms(std::vector<ObbResult>* result, float iou_threshold) {
        const size_t N = result->size();
        std::vector<int> sorted_indices(N);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
            return (*result)[a].score > (*result)[b].score;
        });
        std::vector<bool> suppressed(N, false);
        std::vector<int> keep;
        for (size_t m = 0; m < N; ++m) {
            const int i = sorted_indices[m];
            if (suppressed[i]) continue;
            keep.push_back(i);
            const auto& b_i = (*result)[i].rotated_box;
            for (size_t n = m + 1; n < N; ++n) {
                const int j = sorted_indices[n];
                if (suppressed[j]) continue;
                const auto& b_j = (*result)[j].rotated_box;
                if (ref_rotated_iou(rotated_rect_to_cv_type(b_i), rotated_rect_to_cv_type(b_j)) > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
        std::vector<ObbResult> out;
        out.reserve(keep.size());
        for (const int idx : keep) out.push_back((*result)[idx]);
        result->swap(out);
    }

    std::vector<ObbResult> make_case_clustered() {
        return {
            {{10, 10, 20, 20, 0}, 0, 0.9f},   // 高置信，保留
            {{12, 12, 20, 20, 0}, 0, 0.8f},   // 与[0]高IoU → 抑制
            {{60, 60, 20, 20, 0}, 1, 0.7f},   // 保留
            {{62, 62, 20, 20, 0}, 1, 0.6f},   // 与[2]高IoU → 抑制
            {{100, 100, 20, 20, 0}, 2, 0.5f}, // 独立，保留
            {{150, 10, 20, 20, 0}, 0, 0.4f},  // 独立，保留
        };
    }

    std::vector<ObbResult> make_case_dense(int seed) {
        // 确定性伪随机密集场景，AABB 早筛必须零误差
        std::vector<ObbResult> cases;
        unsigned s = static_cast<unsigned>(seed) * 2654435761u + 12345u;
        auto rnd = [&s]() { s = s * 1103515245u + 12345u; return static_cast<float>((s >> 16) & 0x7fff) / 32767.0f; };
        for (int i = 0; i < 400; ++i) {
            const float x = rnd() * 200.0f;
            const float y = rnd() * 200.0f;
            const float w = 10.0f + rnd() * 30.0f;
            const float h = 10.0f + rnd() * 30.0f;
            const float ang = (rnd() - 0.5f) * 180.0f;
            cases.push_back({{x, y, w, h, ang}, i % 5, rnd()});
        }
        return cases;
    }
} // namespace

TEST_CASE("obb_nms AABB-optimized matches reference (exact equivalence)", "[core]") {
    const float thr = 0.5f;
    const std::vector<std::vector<ObbResult>> cases = {
        make_case_clustered(),
        make_case_dense(1),
        make_case_dense(2),
        make_case_dense(7),
    };
    for (auto base : cases) {
        auto ref = base;
        auto opt = base;
        reference_obb_nms(&ref, thr);
        obb_nms(&opt, thr);
        REQUIRE(ref.size() == opt.size());
        for (size_t i = 0; i < ref.size(); ++i) {
            REQUIRE(ref[i].score == opt[i].score);
            REQUIRE(ref[i].label_id == opt[i].label_id);
            REQUIRE(ref[i].rotated_box.xc == Catch::Approx(opt[i].rotated_box.xc).margin(1e-3f));
            REQUIRE(ref[i].rotated_box.yc == Catch::Approx(opt[i].rotated_box.yc).margin(1e-3f));
            REQUIRE(ref[i].rotated_box.width == Catch::Approx(opt[i].rotated_box.width).margin(1e-3f));
            REQUIRE(ref[i].rotated_box.height == Catch::Approx(opt[i].rotated_box.height).margin(1e-3f));
            REQUIRE(ref[i].rotated_box.angle == Catch::Approx(opt[i].rotated_box.angle).margin(1e-3f));
        }
    }
}
```

- [ ] **Step 2: 在 `tests/CMakeLists.txt` 注册新测试文件**

在 `TEST_SOURCES` 列表中加入一行（放在 `test_vision_models.cpp` 之后）：

```cmake
    test_vision_models.cpp
    test_obb_nms.cpp
```

- [ ] **Step 3: 确认 `ObbResult`/`RotatedRect` 结构与声明确认**

`RotatedRect{xc,yc,width,height,angle}`（float）与 `ObbResult{rotated_box, label_id, score}` 与 `tests/test_vision_models.cpp` 中 `ObbResult` 使用一致；若 `RotatedRect`/`ObbResult` 字段名或聚合顺序不同，按 `csrc/vision/common/struct.h` 实际定义调整 Step 1 测试里的聚合初始化（不改库代码）。确认 `utils.h` 已声明 `obb_nms`（`csrc/vision/utils.h:47`）与 `rotated_rect_to_cv_type`（`csrc/vision/utils.h:38`）。

- [ ] **Step 4: 先构建并运行新测试，确认识别到（当前代码即旧算法，新旧一致必通过）**

Run:
```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && ninja -C build test_modeldeploy'
cd build && ./bin/test_modeldeploy.exe "obb_nms AABB-optimized*" -s
```
Expected: PASS（当前实现 == 参考实现）。

- [ ] **Step 5: 重写 `obb_nms` 实现（`csrc/vision/obb_nms.cpp`）**

将 `utils::obb_nms` 函数体（原 `obb_nms.cpp:118-158`）整体替换为：

```cpp
    void obb_nms(std::vector<ObbResult>* result, const float iou_threshold, std::vector<int>* index) {
        const size_t N = result->size();
        if (N == 0) return;
        // Step 1: 根据分数排序得到索引
        std::vector<int> sorted_indices(N);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](const int a, const int b) {
            return (*result)[a].score > (*result)[b].score; // 分数高的排前面
        });

        // 预计算每个框的 cv::RotatedRect 与其轴对齐外接矩形（AABB），避免内层重复转换与相交开销
        std::vector<cv::RotatedRect> cv_boxes(N);
        std::vector<cv::Rect2f> aabb(N);
        for (size_t m = 0; m < N; ++m) {
            cv_boxes[m] = rotated_rect_to_cv_type((*result)[sorted_indices[m]].rotated_box);
            aabb[m] = cv_boxes[m].boundingRect2f();
        }

        // Step 2: NMS 主逻辑
        std::vector<bool> suppressed(N, false);
        std::vector<int> keep_indices;
        for (size_t m = 0; m < N; ++m) {
            if (suppressed[m]) continue;
            keep_indices.push_back(sorted_indices[m]); // 保留当前框
            const cv::RotatedRect& box_i = cv_boxes[m];
            const cv::Rect2f& r_i = aabb[m];
            for (size_t n = m + 1; n < N; ++n) {
                if (suppressed[n]) continue;
                // AABB 早筛：外接矩形不相交 → 旋转IoU必为0，跳过昂贵的多边形相交（等价变换，不改变抑制决策）
                const cv::Rect2f& r_j = aabb[n];
                if (!(r_i.x < r_j.x + r_j.width && r_j.x < r_i.x + r_i.width &&
                      r_i.y < r_j.y + r_j.height && r_j.y < r_i.y + r_i.height)) {
                    continue;
                }
                if (rotated_iou(box_i, cv_boxes[n]) > iou_threshold) {
                    suppressed[n] = true;
                }
            }
        }

        // Step 3: 根据 keep_indices 重建结果
        std::vector<ObbResult> new_result;
        new_result.reserve(keep_indices.size());
        for (const auto idx : keep_indices) {
            new_result.push_back(std::move((*result)[idx])); // 移动语义
            if (index) {
                index->push_back(idx);
            }
        }
        result->swap(new_result);
    }
```

- [ ] **Step 6: 构建并运行等价性测试**

Run:
```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && ninja -C build test_modeldeploy'
cd build && ./bin/test_modeldeploy.exe "obb_nms AABB-optimized*" -s
```
Expected: PASS。

- [ ] **Step 7: 运行现有 OBB 回归（`[vision_models]`），确认真实模型输出不变**

Run:
```bash
cd build && ./bin/test_modeldeploy.exe "UltralyticsObb model" -s
```
Expected: PASS（`results.size() > 0`，坐标/score 合法）。

- [ ] **Step 8: 运行完整核心测试集，确认无回归**

Run:
```bash
cd build && ./bin/test_modeldeploy.exe [core] -s
```
Expected: 全部 PASS。

- [ ] **Step 9: Commit**

```bash
git add tests/test_obb_nms.cpp tests/CMakeLists.txt csrc/vision/obb_nms.cpp
git commit -m "perf(obb): AABB pre-check + hoist cv conversion in obb_nms (exact-equivalent output)"
```

---

### Task 2: 新增 with-NMS(end2end) OBB 验证用例

**Files:**
- Modify: `benchmark/benchmark_models.cpp`（文件末尾追加一个 `TEST_CASE`）
- Test: `build/bin/benchmark.exe "Benchmark UltralyticsObb end2end*"`

**Interfaces:**
- Consumes: `bench_data_dir()`、`bench_rel(const char*, const char*, OpBackend)`（`benchmark_models.cpp:145`）、`has_file`、`bench_opt(const BenchSpec&, int)`（`benchmark_models.cpp:162`）、`load_img`、`report(const std::string&, const std::vector<TimerArray>&)`；`detection::UltralyticsObb`（`csrc/vision/obb/ultralytics_obb.h`，含 `get_preprocessor().set_size(...)`）。
- Produces: 终端打印 `yolo26n-obb-end2end ORT-CPU | pre/infer/post` 与 `boxes=N`。

- [ ] **Step 1: 在 `benchmark/benchmark_models.cpp` 末尾追加 end2end 用例**

在文件末尾（`TEST_CASE("Benchmark UltralyticsSem", ...)` 等之后）追加如下代码。注意它**只**跑 ORT-CPU，不走 `bench_yolo` 多后端循环（避免依赖不存在的 `mnn/trt` end2end 产物）：

```cpp
// with-NMS(end2end) 模型：输出 [1,300,7]，模型内已做 NMS，SDK 走 run_with_nms（post≈0）。
// end2end 输入为 1024x1024，预处理器默认 640 需显式覆盖。仅 ORT-CPU 验证。
TEST_CASE("Benchmark UltralyticsObb end2end (with-NMS)", "[all_models][benchmark]") {
    constexpr int kRuns = 20;
    const auto rel = bench_rel("yolo26n", "yolo26n-obb-end2end", OpBackend::OrtCpu);
    const auto mp = bench_data_dir() / "test_models" / rel;
    if (!has_file(mp)) {
        std::printf("[bench][skip] ORT-CPU %s (missing)\n", rel.string().c_str());
        return;
    }
    try {
        RuntimeOption opt = bench_opt(BenchSpec{OpBackend::OrtCpu, "ORT-CPU"}, 1024);
        detection::UltralyticsObb model(mp.string(), opt);
        if (!model.is_initialized()) {
            std::printf("[bench][skip] ORT-CPU %s (init fail)\n", rel.string().c_str());
            return;
        }
        model.get_preprocessor().set_size({1024, 1024});
        auto img = load_img("test_obb.jpg");
        if (img.empty()) return;

        std::vector<ObbResult> result;
        if (!model.predict(img, &result)) {
            std::printf("[bench][error] yolo26n-obb-end2end predict failed\n");
            return;
        }
        std::printf("[bench] yolo26n-obb-end2end boxes=%zu\n", result.size());

        for (int i = 0; i < 5; ++i) model.predict(img, &result);  // 预热
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            TimerArray t;
            if (!model.predict(img, &result, &t)) { runs.clear(); break; }
            runs.push_back(t);
        }
        if (!runs.empty()) report("yolo26n-obb-end2end ORT-CPU", runs);
    } catch (const std::exception& e) {
        std::printf("[bench][error] yolo26n-obb-end2end (%s)\n", e.what());
    } catch (...) {
        std::printf("[bench][error] yolo26n-obb-end2end (unknown exception)\n");
    }
}
```

- [ ] **Step 2: 构建 benchmark**

Run:
```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && ninja -C build benchmark'
```
Expected: 无编译错误。

- [ ] **Step 3: 运行 end2end 用例，确认 post≈0 且 boxes>0**

Run:
```bash
cd build && $env:TEST_DATA_DIR="E:\CLionProjects\ModelDeploy" ; ./bin/benchmark.exe "Benchmark UltralyticsObb end2end*"
```
Expected:
```text
[bench] yolo26n-obb-end2end boxes=<N>       (N > 0)
[bench] yolo26n-obb-end2end ORT-CPU | pre=...ms infer=...ms post=<~0>ms total=...ms (n=20)
```
其中 `post` 应接近 0（NMS 在模型内）。

- [ ] **Step 4: 跑一遍完整 benchmark 确认无回归**

Run:
```bash
cd build && $env:TEST_DATA_DIR="E:\CLionProjects\ModelDeploy" ; ./bin/benchmark.exe
```
Expected: 全部用例 PASS（含原有 18 用例）+ 新增 end2end 用例。

- [ ] **Step 5: 对比 no-NMS obb 优化前后 post 耗时（人工核验）**

Run:
```bash
cd build && $env:TEST_DATA_DIR="E:\CLionProjects\ModelDeploy" ; ./bin/benchmark.exe "Benchmark UltralyticsObb"
```
Expected: `yolo26n-obb`（no-NMS, ORT-CPU）在 `test_obb.jpg` 上的 `post` 相比优化前（~10.7ms）显著下降，且 `TRT-engine` 的 obb post（~8.7ms）明显下降。

- [ ] **Step 6: Commit**

```bash
git add benchmark/benchmark_models.cpp
git commit -m "bench(obb): add with-NMS(end2end) UltralyticsObb validation case (ORT-CPU, 1024)"
```

---

## Self-Review

**Spec coverage:**
- 方案 A（优化 obb_nms，AABB 预筛 + 提拉 cv 转换，结果逐位不变）→ Task 1，含等价性单元测试 + 现有 `[vision_models]` 回归。
- 方案 B（with-NMS end2end，ORT-CPU，`set_size({1024,1024})`，post≈0）→ Task 2。
- 方案 C（回归/对比验证）→ Task 1 Step 6-8、Task 2 Step 3-5。

**Placeholder scan:** 无 TBD/TODO；每步含完整代码与命令。

**Type consistency:** `RotatedRect`/`ObbResult` 聚合初始化需按 `struct.h` 实际定义核对（Step 3 明确列出该依赖）；`obb_nms`/`rotated_rect_to_cv_type` 均以 `utils.h` 声明为准；Task 2 全部依赖（`bench_rel`/`bench_opt`/`BenchSpec`/`report`/`UltralyticsObb::get_preprocessor().set_size`）与 `benchmark_models.cpp` 现有定义一致。
