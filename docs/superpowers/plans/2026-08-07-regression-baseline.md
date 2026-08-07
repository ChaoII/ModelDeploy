# 回归基线测试系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为每个模型建立输出基线（预处理张量/原始推理/后处理结果），修改后 ctest 自动对比防回归。

**Architecture:** 三个组件：`baseline_utils`（JSON 序列化/反序列化/浮动阈值对比，基于 nlohmann/json）、`baseline_collect`（独立 exe，加载模型跑图生成基线 JSON 文件）、`baseline_compare`（Catch2 测试 `[regression]` 标签，重新跑图与基线对比）。基线文件存 `tests/baselines/`，git 管理。

**Tech Stack:** C++17、nlohmann/json 3.11.3（`application/third_party/nlohmann/json.hpp`）、Catch2 v3、ModelDeploySDK、OnnxRuntime(CPU)

## Global Constraints

- JSON 库：复用 `application/third_party/nlohmann/json.hpp`（单头文件 3.11.3），不引入新依赖
- 所有数值以 `%.6f` 截断存储（FP32 6 位小数）
- 阈值：Detection/Obb/Pose/OCR-det 坐标 ±1px，Obb angle ±0.5°，score ±0.01，label 严格相等，Seg mask 差异像素比例 < 0.1%，Tensor/预处理数值 ±1e-5，Tensor shape 严格相等，OCR rec text 严格相等
- 模型/图片/基线缺失时 `return` skip（不 FAIL），与现有测试一致
- 测试源文件需手动加入 `tests/CMakeLists.txt` 的 `TEST_SOURCES`（显式列表，非 GLOB）
- MSVC 需 `/utf-8`（根 CMakeLists 已为 SDK 设置，新增测试文件同样生效）
- 结果结构：`Rect2f{x,y,width,height}`、`RotatedRect{xc,yc,width,height,angle}`、`Point3f{x,y,z}`、`ObbResult{rotated_box,label_id,score}`、`DetectionResult{box,label_id,score}`、`InstanceSegResult{box,mask,label_id,score}`、`KeyPointsResult{box,keypoints,label_id,score}`、`ClassifyResult{label_ids,scores}`、`OCRResult{boxes,text,rec_scores,cls_scores,cls_labels}`

---

### Task 1: baseline_utils — 序列化/反序列化/对比

**Files:**
- Create: `tests/baseline_utils.h`
- Create: `tests/baseline_utils.cpp`

**Interfaces:**
- Produces:
  - `json serialize_tensor(const Tensor& t)` — 存 `{name, shape, dtype, values(截断前10000), stats{min,max,mean}}`
  - `json serialize_detection(const std::vector<DetectionResult>&)`
  - `json serialize_obb(const std::vector<ObbResult>&)`
  - `json serialize_seg(const std::vector<InstanceSegResult>&)`
  - `json serialize_pose(const std::vector<KeyPointsResult>&)`
  - `json serialize_cls(const ClassifyResult&)`
  - `json serialize_ocr_det(const std::vector<std::array<int,8>>&)`
  - `json serialize_ocr_rec(const std::string& text, float score)`
  - `json serialize_ocr_cls(int32_t label, float score)`
  - `std::vector<std::string> compare_tensor(const json& baseline, const Tensor& current)` — 空向量=通过
  - `std::vector<std::string> compare_detection(const json&, const std::vector<DetectionResult>&)`
  - `std::vector<std::string> compare_obb(const json&, const std::vector<ObbResult>&)`
  - `std::vector<std::string> compare_seg(const json&, const std::vector<InstanceSegResult>&)`
  - `std::vector<std::string> compare_pose(const json&, const std::vector<KeyPointsResult>&)`
  - `std::vector<std::string> compare_cls(const json&, const ClassifyResult&)`
  - `std::vector<std::string> compare_ocr_det(const json&, const std::vector<std::array<int,8>>&)`
  - `std::vector<std::string> compare_ocr_rec(const json&, const std::string&, float)`
  - `std::vector<std::string> compare_ocr_cls(const json&, int32_t, float)`
  - `std::vector<std::string> compare_counts(const json&, size_t current_count)` — 校验实例数
  - `std::string dtype_to_str(DataType)` / 内部辅助

- [ ] **Step 1: 创建 baseline_utils.h**

```cpp
#pragma once
#include <vector>
#include <array>
#include <string>
#include "csrc/vision/common/result.h"
#include "csrc/core/tensor.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace modeldeploy::vision::baseline {

    json serialize_tensor(const Tensor& t);

    json serialize_detection(const std::vector<DetectionResult>& rs);
    json serialize_obb(const std::vector<ObbResult>& rs);
    json serialize_seg(const std::vector<InstanceSegResult>& rs);
    json serialize_pose(const std::vector<KeyPointsResult>& rs);
    json serialize_cls(const ClassifyResult& r);
    json serialize_ocr_det(const std::vector<std::array<int, 8>>& boxes);
    json serialize_ocr_rec(const std::string& text, float score);
    json serialize_ocr_cls(int32_t label, float score);

    std::vector<std::string> compare_tensor(const json& base, const Tensor& cur);
    std::vector<std::string> compare_detection(const json& base, const std::vector<DetectionResult>& rs);
    std::vector<std::string> compare_obb(const json& base, const std::vector<ObbResult>& rs);
    std::vector<std::string> compare_seg(const json& base, const std::vector<InstanceSegResult>& rs);
    std::vector<std::string> compare_pose(const json& base, const std::vector<KeyPointsResult>& rs);
    std::vector<std::string> compare_cls(const json& base, const ClassifyResult& r);
    std::vector<std::string> compare_ocr_det(const json& base, const std::vector<std::array<int, 8>>& boxes);
    std::vector<std::string> compare_ocr_rec(const json& base, const std::string& text, float score);
    std::vector<std::string> compare_ocr_cls(const json& base, int32_t label, float score);
    std::vector<std::string> compare_counts(const json& base, size_t cur_count);

    std::string dtype_to_str(DataType dtype);
}
```

- [ ] **Step 2: 实现 baseline_utils.cpp**

```cpp
#include "baseline_utils.h"
#include <cmath>
#include <cstdio>

namespace modeldeploy::vision::baseline {

    static constexpr float COORD_TOL = 1.0f;   // px
    static constexpr float ANGLE_TOL = 0.5f;   // degrees
    static constexpr float SCORE_TOL = 0.01f;
    static constexpr float TENSOR_TOL = 1e-5f;
    static constexpr size_t MAX_TENSOR_ELEMS = 10000;

    std::string dtype_to_str(DataType dtype) {
        switch (dtype) {
            case DataType::FP32: return "FP32";
            case DataType::FP16: return "FP16";
            case DataType::INT32: return "INT32";
            case DataType::INT64: return "INT64";
            case DataType::UINT8: return "UINT8";
            default: return "UNKNOWN";
        }
    }

    static std::string fmt(const char* fmt, float v) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), fmt, v);
        return buf;
    }

    static bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

    json serialize_tensor(const Tensor& t) {
        json j;
        j["name"] = t.get_name();
        j["shape"] = t.shape();
        j["dtype"] = dtype_to_str(t.dtype());
        const size_t n = t.size();
        const float* d = static_cast<const float*>(t.data());
        std::vector<float> vals;
        vals.reserve(std::min(n, MAX_TENSOR_ELEMS));
        float mn = 0, mx = 0, sum = 0;
        if (n > 0) { mn = mx = d[0]; }
        for (size_t i = 0; i < n && i < MAX_TENSOR_ELEMS; ++i) {
            float v = d[i];
            vals.push_back(std::roundf(v * 1e6f) / 1e6f);
            if (i < n) { if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
        }
        // 统计遍历全部
        mn = d[0]; mx = d[0]; sum = 0;
        for (size_t i = 0; i < n; ++i) {
            float v = d[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        j["values"] = vals;
        j["stats"] = { {"min", std::roundf(mn * 1e6f) / 1e6f},
                       {"max", std::roundf(mx * 1e6f) / 1e6f},
                       {"mean", std::roundf((sum / n) * 1e6f) / 1e6f} };
        j["numel"] = n;
        return j;
    }

    static json serialize_box(const Rect2f& b) {
        return json{ {"x", std::roundf(b.x * 1e6f) / 1e6f},
                     {"y", std::roundf(b.y * 1e6f) / 1e6f},
                     {"width", std::roundf(b.width * 1e6f) / 1e6f},
                     {"height", std::roundf(b.height * 1e6f) / 1e6f} };
    }

    json serialize_detection(const std::vector<DetectionResult>& rs) {
        json arr = json::array();
        for (auto& r : rs)
            arr.push_back({ {"box", serialize_box(r.box)}, {"label_id", r.label_id},
                            {"score", std::roundf(r.score * 1e6f) / 1e6f} });
        return arr;
    }

    json serialize_obb(const std::vector<ObbResult>& rs) {
        json arr = json::array();
        for (auto& r : rs)
            arr.push_back({ {"xc", std::roundf(r.rotated_box.xc * 1e6f) / 1e6f},
                            {"yc", std::roundf(r.rotated_box.yc * 1e6f) / 1e6f},
                            {"width", std::roundf(r.rotated_box.width * 1e6f) / 1e6f},
                            {"height", std::roundf(r.rotated_box.height * 1e6f) / 1e6f},
                            {"angle", std::roundf(r.rotated_box.angle * 1e6f) / 1e6f},
                            {"label_id", r.label_id},
                            {"score", std::roundf(r.score * 1e6f) / 1e6f} });
        return arr;
    }

    json serialize_seg(const std::vector<InstanceSegResult>& rs) {
        json arr = json::array();
        for (auto& r : rs) {
            json mask = { {"w", (int)r.mask.shape.size() > 1 ? (long long)r.mask.shape[1] : 0LL},
                          {"h", (int)r.mask.shape.size() > 0 ? (long long)r.mask.shape[0] : 0LL} };
            const uint8_t* md = r.mask.buffer.empty() ? nullptr : r.mask.buffer.data();
            if (md && r.mask.shape.size() == 2) {
                size_t npix = (size_t)r.mask.shape[0] * (size_t)r.mask.shape[1];
                size_t cnt = 0;
                for (size_t i = 0; i < npix; ++i) if (md[i] > 0) ++cnt;
                mask["nonzero_ratio"] = std::roundf((float)cnt / npix * 1e6f) / 1e6f;
            }
            arr.push_back({ {"box", serialize_box(r.box)}, {"label_id", r.label_id},
                            {"score", std::roundf(r.score * 1e6f) / 1e6f}, {"mask", mask} });
        }
        return arr;
    }

    json serialize_pose(const std::vector<KeyPointsResult>& rs) {
        json arr = json::array();
        for (auto& r : rs) {
            json kps = json::array();
            for (auto& k : r.keypoints)
                kps.push_back({ std::roundf(k.x * 1e6f) / 1e6f,
                                std::roundf(k.y * 1e6f) / 1e6f,
                                std::roundf(k.z * 1e6f) / 1e6f });
            arr.push_back({ {"box", serialize_box(r.box)}, {"label_id", r.label_id},
                            {"score", std::roundf(r.score * 1e6f) / 1e6f}, {"keypoints", kps} });
        }
        return arr;
    }

    json serialize_cls(const ClassifyResult& r) {
        json j;
        j["label_ids"] = r.label_ids;
        j["scores"] = json::array();
        for (auto s : r.scores) j["scores"].push_back(std::roundf(s * 1e6f) / 1e6f);
        return j;
    }

    json serialize_ocr_det(const std::vector<std::array<int, 8>>& boxes) {
        json arr = json::array();
        for (auto& b : boxes) arr.push_back(b);
        return arr;
    }

    json serialize_ocr_rec(const std::string& text, float score) {
        return json{ {"text", text}, {"score", std::roundf(score * 1e6f) / 1e6f} };
    }

    json serialize_ocr_cls(int32_t label, float score) {
        return json{ {"label", label}, {"score", std::roundf(score * 1e6f) / 1e6f} };
    }

    std::vector<std::string> compare_counts(const json& base, size_t cur_count) {
        std::vector<std::string> diffs;
        size_t base_count = base.size();
        if (base_count != cur_count)
            diffs.push_back("instance count mismatch: baseline=" + std::to_string(base_count) +
                            " current=" + std::to_string(cur_count));
        return diffs;
    }

    static void check_coord(const char* field, float base, float cur, std::vector<std::string>* diffs) {
        if (!near(base, cur, COORD_TOL))
            diffs->push_back(fmt("field[%s] mismatch: base=%.6f cur=%.6f", field, base, cur));
    }
    static void check_score(const char* field, float base, float cur, std::vector<std::string>* diffs) {
        if (!near(base, cur, SCORE_TOL))
            diffs->push_back(fmt("field[%s] mismatch: base=%.6f cur=%.6f", field, base, cur));
    }
    static void check_angle(float base, float cur, std::vector<std::string>* diffs) {
        if (!near(base, cur, ANGLE_TOL))
            diffs->push_back(fmt("field[angle] mismatch: base=%.6f cur=%.6f", base, cur));
    }

    std::vector<std::string> compare_detection(const json& base, const std::vector<DetectionResult>& rs) {
        auto diffs = compare_counts(base, rs.size());
        if (!diffs.empty()) return diffs;
        for (size_t i = 0; i < rs.size(); ++i) {
            const auto& b = base[i];
            check_coord("x", b["box"]["x"], rs[i].box.x, &diffs);
            check_coord("y", b["box"]["y"], rs[i].box.y, &diffs);
            check_coord("width", b["box"]["width"], rs[i].box.width, &diffs);
            check_coord("height", b["box"]["height"], rs[i].box.height, &diffs);
            if ((int)b["label_id"] != rs[i].label_id)
                diffs.push_back(fmt("instance[%zu] label mismatch: base=%d cur=%d", i, (int)b["label_id"], rs[i].label_id));
            check_score("score", b["score"], rs[i].score, &diffs);
        }
        return diffs;
    }

    std::vector<std::string> compare_obb(const json& base, const std::vector<ObbResult>& rs) {
        auto diffs = compare_counts(base, rs.size());
        if (!diffs.empty()) return diffs;
        for (size_t i = 0; i < rs.size(); ++i) {
            const auto& b = base[i];
            check_coord("xc", b["xc"], rs[i].rotated_box.xc, &diffs);
            check_coord("yc", b["yc"], rs[i].rotated_box.yc, &diffs);
            check_coord("width", b["width"], rs[i].rotated_box.width, &diffs);
            check_coord("height", b["height"], rs[i].rotated_box.height, &diffs);
            check_angle(b["angle"], rs[i].rotated_box.angle, &diffs);
            if ((int)b["label_id"] != rs[i].label_id)
                diffs.push_back(fmt("instance[%zu] label mismatch: base=%d cur=%d", i, (int)b["label_id"], rs[i].label_id));
            check_score("score", b["score"], rs[i].score, &diffs);
        }
        return diffs;
    }

    std::vector<std::string> compare_seg(const json& base, const std::vector<InstanceSegResult>& rs) {
        auto diffs = compare_counts(base, rs.size());
        if (!diffs.empty()) return diffs;
        for (size_t i = 0; i < rs.size(); ++i) {
            const auto& b = base[i];
            check_coord("x", b["box"]["x"], rs[i].box.x, &diffs);
            check_coord("y", b["box"]["y"], rs[i].box.y, &diffs);
            check_coord("width", b["box"]["width"], rs[i].box.width, &diffs);
            check_coord("height", b["box"]["height"], rs[i].box.height, &diffs);
            if ((int)b["label_id"] != rs[i].label_id)
                diffs.push_back(fmt("instance[%zu] label mismatch: base=%d cur=%d", i, (int)b["label_id"], rs[i].label_id));
            check_score("score", b["score"], rs[i].score, &diffs);
            if (b["mask"].contains("nonzero_ratio") && rs[i].mask.shape.size() == 2) {
                float base_ratio = b["mask"]["nonzero_ratio"];
                size_t npix = (size_t)rs[i].mask.shape[0] * (size_t)rs[i].mask.shape[1];
                size_t cnt = 0;
                for (size_t k = 0; k < npix; ++k) if (rs[i].mask.buffer[k] > 0) ++cnt;
                float cur_ratio = npix ? (float)cnt / npix : 0.f;
                if (std::fabs(base_ratio - cur_ratio) > 0.001f)
                    diffs.push_back(fmt("instance[%zu] mask nonzero_ratio mismatch: base=%.6f cur=%.6f",
                                        i, base_ratio, cur_ratio));
            }
        }
        return diffs;
    }

    std::vector<std::string> compare_pose(const json& base, const std::vector<KeyPointsResult>& rs) {
        auto diffs = compare_counts(base, rs.size());
        if (!diffs.empty()) return diffs;
        for (size_t i = 0; i < rs.size(); ++i) {
            const auto& b = base[i];
            check_coord("x", b["box"]["x"], rs[i].box.x, &diffs);
            check_coord("y", b["box"]["y"], rs[i].box.y, &diffs);
            check_coord("width", b["box"]["width"], rs[i].box.width, &diffs);
            check_coord("height", b["box"]["height"], rs[i].box.height, &diffs);
            if ((int)b["label_id"] != rs[i].label_id)
                diffs.push_back(fmt("instance[%zu] label mismatch: base=%d cur=%d", i, (int)b["label_id"], rs[i].label_id));
            if (b["keypoints"].size() != rs[i].keypoints.size()) {
                diffs.push_back(fmt("instance[%zu] keypoint count mismatch", i));
            } else {
                for (size_t k = 0; k < rs[i].keypoints.size(); ++k) {
                    check_coord("kp_x", b["keypoints"][k][0], rs[i].keypoints[k].x, &diffs);
                    check_coord("kp_y", b["keypoints"][k][1], rs[i].keypoints[k].y, &diffs);
                }
            }
        }
        return diffs;
    }

    std::vector<std::string> compare_cls(const json& base, const ClassifyResult& r) {
        std::vector<std::string> diffs;
        if (base["label_ids"].size() != r.label_ids.size())
            return {"label count mismatch"};
        for (size_t i = 0; i < r.label_ids.size(); ++i) {
            if ((int)base["label_ids"][i] != r.label_ids[i])
                diffs.push_back(fmt("label_ids[%zu] mismatch: base=%d cur=%d", i, (int)base["label_ids"][i], r.label_ids[i]));
            if (!near(base["scores"][i], r.scores[i], SCORE_TOL))
                diffs.push_back(fmt("scores[%zu] mismatch: base=%.6f cur=%.6f", i, (float)base["scores"][i], r.scores[i]));
        }
        return diffs;
    }

    std::vector<std::string> compare_ocr_det(const json& base, const std::vector<std::array<int, 8>>& boxes) {
        auto diffs = compare_counts(base, boxes.size());
        if (!diffs.empty()) return diffs;
        for (size_t i = 0; i < boxes.size(); ++i)
            for (int k = 0; k < 8; ++k)
                if (std::abs((int)base[i][k] - boxes[i][k]) > COORD_TOL)
                    diffs.push_back(fmt("box[%zu][%d] mismatch: base=%d cur=%d", i, k, (int)base[i][k], boxes[i][k]));
        return diffs;
    }

    std::vector<std::string> compare_ocr_rec(const json& base, const std::string& text, float score) {
        std::vector<std::string> diffs;
        if (base["text"].get<std::string>() != text)
            diffs.push_back("text mismatch: base=[" + base["text"].get<std::string>() + "] cur=[" + text + "]");
        if (!near(base["score"], score, SCORE_TOL))
            diffs.push_back(fmt("score mismatch: base=%.6f cur=%.6f", (float)base["score"], score));
        return diffs;
    }

    std::vector<std::string> compare_ocr_cls(const json& base, int32_t label, float score) {
        std::vector<std::string> diffs;
        if ((int)base["label"] != label)
            diffs.push_back(fmt("label mismatch: base=%d cur=%d", (int)base["label"], label));
        if (!near(base["score"], score, SCORE_TOL))
            diffs.push_back(fmt("score mismatch: base=%.6f cur=%.6f", (float)base["score"], score));
        return diffs;
    }

    std::vector<std::string> compare_tensor(const json& base, const Tensor& cur) {
        std::vector<std::string> diffs;
        if (base["shape"] != json(cur.shape()))
            return {"tensor shape mismatch"};
        if (base["dtype"].get<std::string>() != dtype_to_str(cur.dtype()))
            return {"tensor dtype mismatch"};
        const size_t n = cur.size();
        const float* d = static_cast<const float*>(cur.data());
        if (base["numel"].get<size_t>() != n)
            return {"tensor numel mismatch"};
        // 统计比较
        float mn = d[0], mx = d[0], sum = 0;
        for (size_t i = 0; i < n; ++i) { if (d[i] < mn) mn = d[i]; if (d[i] > mx) mx = d[i]; sum += d[i]; }
        if (std::fabs((float)base["stats"]["min"] - mn) > TENSOR_TOL) diffs.push_back("tensor min mismatch");
        if (std::fabs((float)base["stats"]["max"] - mx) > TENSOR_TOL) diffs.push_back("tensor max mismatch");
        if (std::fabs((float)base["stats"]["mean"] - sum / n) > TENSOR_TOL) diffs.push_back("tensor mean mismatch");
        // 抽样比较
        size_t m = std::min(n, MAX_TENSOR_ELEMS);
        for (size_t i = 0; i < m; ++i) {
            float bv = base["values"][i];
            if (std::fabs(bv - d[i]) > TENSOR_TOL) {
                diffs.push_back(fmt("tensor[%zu] mismatch: base=%.6f cur=%.6f", i, bv, d[i]));
                if (diffs.size() >= 20) break;
            }
        }
        return diffs;
    }
}
```

- [ ] **Step 3: 编译验证（无测试）**

Run: `cmake --build build --target test_modeldeploy`（暂未加入 CMake，跳过；改为单独编译 baseline_utils.cpp 验证语法）
Run: `cl /nologo /std:c++17 /utf-8 /EHsc /I. /Iapplication\third_party /c tests\baseline_utils.cpp /Fo:build\obj\baseline_utils.obj`
Expected: 无错误（.obj 生成）

- [ ] **Step 4: Commit**

```bash
git add tests/baseline_utils.h tests/baseline_utils.cpp
git commit -m "test: 新增回归基线序列化/对比工具 baseline_utils"
```

---

### Task 2: baseline_collect — 独立收集器

**Files:**
- Create: `tests/baseline_collect.cpp`

**Interfaces:**
- Consumes: `baseline_utils.h` 的 `serialize_*` 函数
- Produces: 独立 exe `baseline_collect.exe`，用法 `--model <path> --image <path> --out <dir>`，按模型类型生成对应 JSON 基线文件
- 用 `modeldeploy::BaseModel` + `predict`/`infer`/`preprocessor.run` 拿三类输出
- 按模型文件名后缀推断类型（`.onnx`/`.mnn`），需用 `--type` 参数或自动探测

- [ ] **Step 1: 实现 baseline_collect.cpp**

要点：
- 支持 `--type det|obb|seg|pose|cls|ocr_det|ocr_rec|ocr_cls` 参数
- 加载模型（CPU ORT），`option.use_cpu()`
- det/obb/seg/pose/cls：`model.predict(img, &results)` → serialize_xxx
- pre/raw：`model.get_preprocessor().run({img}, &inputs, &lbs)` → serialize_tensor(each input)；`model.infer(inputs, &outputs)` → serialize_tensor(each output)
- ocr_det：`DBDetector.predict(img, &OCRResult)` 取 `ocr_result.boxes`
- ocr_rec：`Recognizer.predict(img, &text, &score)`（需 dict 路径）
- ocr_cls：`Classifier.predict(img, &label, &score)`
- 输出文件 `<out>/<model文件名>.<type>.json`，含 meta

基线文件顶层约定（对比器依赖）：
- 后处理结果：`{ "meta": {...}, "results": <serialize_xxx_result 数组> }`
- 张量：`{ "meta": {...}, "tensor": <serialize_tensor 返回的对象> }`

- [ ] **Step 2: 实现 CLI 解析**

```cpp
// 简化的参数解析
struct Args { std::string model, image, out, type; };
static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--model" && i + 1 < argc) a.model = argv[++i];
        else if (k == "--image" && i + 1 < argc) a.image = argv[++i];
        else if (k == "--out" && i + 1 < argc) a.out = argv[++i];
        else if (k == "--type" && i + 1 < argc) a.type = argv[++i];
    }
    return a;
}
```

- [ ] **Step 3: 编译并生成 yolo11n 检测基线**

```bash
cmake --build build --target baseline_collect
./build/bin/baseline_collect.exe --model test_data/test_models/yolo11n.onnx --image test_data/test_images/test_detection0.jpg --out tests/baselines --type det
./build/bin/baseline_collect.exe --model test_data/test_models/yolo11n.onnx --image test_data/test_images/test_detection0.jpg --out tests/baselines --type pre
./build/bin/baseline_collect.exe --model test_data/test_models/yolo11n.onnx --image test_data/test_images/test_detection0.jpg --out tests/baselines --type raw
```
Expected: `tests/baselines/yolo11n.onnx.det.json`、`.pre.json`、`.raw.json` 生成，内容可读

- [ ] **Step 4: Commit**

```bash
git add tests/baseline_collect.cpp tests/CMakeLists.txt
git commit -m "feat: 新增回归基线收集器 baseline_collect"
```

---

### Task 3: baseline_compare — Catch2 回归对比器

**Files:**
- Create: `tests/baseline_compare.cpp`

**Interfaces:**
- Consumes: `baseline_utils.h` 的 `compare_*` 函数
- Produces: Catch2 测试用例，标签 `[regression]`，模型/图片/基线缺失即 `return`
- 覆盖模型：yolo11n(det/pre/raw)、yolo11n_nms(det)、yolo11n-seg(seg)、yolo11n-pose(pose)、yolo11n-obb/obb_nms(obb)、yolo11n-cls(cls)、scrfd(face det)、ppocrv4 det/rec/cls

- [ ] **Step 1: 实现 baseline_compare.cpp**

```cpp
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include "baseline_utils.h"
#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

namespace fs = std::filesystem;
using namespace modeldeploy;
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::baseline;

static fs::path get_test_data() {
    const char* env = std::getenv("TEST_DATA_DIR");
    if (env && *env) return fs::path(env) / "test_data";
    return fs::current_path() / "test_data";
}
static fs::path baseline_dir() { return get_test_data().parent_path() / "tests" / "baselines"; }
static fs::path model_path(const std::string& rel) { return get_test_data() / "test_models" / rel; }
static fs::path image_path(const std::string& name) { return get_test_data() / "test_images" / name; }

static json load_json(const fs::path& p) {
    std::ifstream f(p);
    json j; f >> j; return j;
}

static RuntimeOption cpu_option() {
    RuntimeOption opt;
    opt.use_cpu();
    opt.set_cpu_thread_num(4);
    return opt;
}

static void require_no_diff(const std::vector<std::string>& diffs) {
    for (auto& d : diffs) FAIL_CHECK(d);
}

static void check_tensors(const std::vector<fs::path>& files,
                          const std::vector<Tensor>& tensors) {
    REQUIRE(files.size() == tensors.size());
    for (size_t i = 0; i < files.size(); ++i) {
        if (!fs::exists(files[i])) return;   // 基线缺失 skip
        auto base = load_json(files[i]);
        require_no_diff(compare_tensor(base["tensor"], tensors[i]));
    }
}
```
对比器约定：基线文件顶层为 `{"meta": {...}, "tensor": {...}}`（单个张量）或 `{"meta": {...}, "results": [...]}`（后处理结果）。`serialize_tensor` 返回的张量 json 由收集器包进 `"tensor"` 键；`serialize_xxx_result` 返回的数组由收集器包进 `"results"` 键。

对每种模型写 TEST_CASE（模板）：

```cpp
template<typename Model, typename Result>
static void collect_outputs(Model& model, const ImageData& img,
                            std::vector<Tensor>* inputs,
                            std::vector<Tensor>* outputs,
                            std::vector<Result>* results) {
    std::vector<LetterBoxRecord> lbs;
    model.get_preprocessor().run({img}, inputs, &lbs);
    model.infer(*inputs, outputs);
    REQUIRE(model.predict(img, results, nullptr));
}
```

TEST_CASE 示例：

```cpp
TEST_CASE("Regression: yolo11n detection", "[regression]") {
    auto modelfile = model_path("yolo11n.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;

    UltralyticsDet model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;

    auto img = ImageData::imread(imgf.string());
    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));

    auto base = load_json(baseline_dir() / "yolo11n.onnx.det.json");
    require_no_diff(compare_detection(base["results"], results));
}
```

每个模型类似，OCR 用专用 predict 接口。

- [ ] **Step 2: 编译并跑通**

```bash
cmake --build build --target test_modeldeploy
./build/bin/test_modeldeploy.exe "[regression]"
```
Expected: 全部通过（基线已由 Task 2 生成）

- [ ] **Step 3: Commit**

```bash
git add tests/baseline_compare.cpp tests/CMakeLists.txt
git commit -m "test: 新增回归基线对比器 baseline_compare"
```

---

### Task 4: CMake 集成

**Files:**
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- 把 `baseline_collect.cpp` 加为独立 exe target（`baseline_collect`）
- 把 `baseline_compare.cpp`、`baseline_utils.cpp` 加入 `TEST_SOURCES`

- [ ] **Step 1: 修改 tests/CMakeLists.txt**

```cmake
set(TEST_SOURCES
    test_core.cpp
    test_bugfix.cpp
    test_image_data.cpp
    test_capi_image.cpp
    test_md_image.cpp
    test_vision_models.cpp
    test_processor_accuracy.cpp
    test_encryption.cpp
    baseline_compare.cpp
    baseline_utils.cpp
    utils.cpp
)

add_executable(test_modeldeploy ${TEST_SOURCES})
target_link_libraries(test_modeldeploy PRIVATE
    ${LIBRARY_NAME}
    Catch2::Catch2WithMain
    ${OpenCV_LIBS})
target_include_directories(test_modeldeploy PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/application/third_party)   # nlohmann json
target_compile_definitions(test_modeldeploy PRIVATE
    MODELDEPLOY_CXX_EXPORT=)

add_executable(baseline_collect baseline_collect.cpp baseline_utils.cpp)
target_link_libraries(baseline_collect PRIVATE
    ${LIBRARY_NAME}
    ${OpenCV_LIBS})
target_include_directories(baseline_collect PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/application/third_party)
target_compile_definitions(baseline_collect PRIVATE
    MODELDEPLOY_CXX_EXPORT=)
```

- [ ] **Step 2: 编译全部**

```bash
cmake -S . -B build -DBUILD_TESTS=ON 2>&1 | Select-String -Pattern "Configuring done|Error"
cmake --build build --target test_modeldeploy baseline_collect
```
Expected: 编译成功，`build/bin/baseline_collect.exe` 和 `build/bin/test_modeldeploy.exe` 生成

- [ ] **Step 3: 验证回归测试集成**

```bash
cd build
ctest -R test_modeldeploy --output-on-failure
```
Expected: `test_modeldeploy` 通过（含 [regression]）

- [ ] **Step 4: Commit**

```bash
git add tests/CMakeLists.txt
git commit -m "build: 回归基线测试集成到 CMake（[regression] 标签 + baseline_collect exe）"
```

---

### Task 5: 生成全部基线 + 灵敏度验证

**Files:**
- Create: `tests/baselines/*.json`（约 16 个文件）

**Interfaces:**
- 用 Task 2 的 `baseline_collect.exe` 为所有模型生成基线
- 用 Task 3 的对比器验证自洽 + 灵敏度

- [ ] **Step 1: 生成全部基线**

```bash
./build/bin/baseline_collect.exe --model test_data/test_models/yolo11n.onnx        --image test_data/test_images/test_detection0.jpg --out tests/baselines --type det
./build/bin/baseline_collect.exe --model test_data/test_models/yolo11n.onnx        --image test_data/test_images/test_detection0.jpg --out tests/baselines --type pre
./build/bin/baseline_collect.exe --model test_data/test_models/yolo11n.onnx        --image test_data/test_images/test_detection0.jpg --out tests/baselines --type raw
./build/bin/baseline_collect.exe --model test_data/test_models/yolo11n_nms.onnx    --image test_data/test_images/test_detection0.jpg --out tests/baselines --type det
./build/bin/baseline_collect.exe --model test_data/test_models/yolo11n-seg.onnx    --image test_data/test_images/test_person.jpg        --out tests/baselines --type seg
./build/bin/baseline_collect.exe --model test_data/test_models/yolo11n-pose.onnx   --image test_data/test_images/test_person.jpg        --out tests/baselines --type pose
./build/bin/baseline_collect.exe --model test_data/test_models/yolo11n-obb.onnx    --image test_data/test_images/test_obb1.jpg          --out tests/baselines --type obb
./build/bin/baseline_collect.exe --model test_data/test_models/yolo11n-obb_nms.onnx --image test_data/test_images/test_obb1.jpg        --out tests/baselines --type obb
./build/bin/baseline_collect.exe --model test_data/test_models/yolo11n-cls.onnx    --image test_data/test_images/test_person.jpg        --out tests/baselines --type cls
./build/bin/baseline_collect.exe --model test_data/test_models/face/scrfd_2.5g_bnkps_shape640x640.onnx --image test_data/test_images/test_face_detection.jpg --out tests/baselines --type face_det
./build/bin/baseline_collect.exe --model test_data/test_models/ocr/ppocrv4_mobile/det_infer.onnx --image test_data/test_images/test_ocr.png --out tests/baselines --type ocr_det
./build/bin/baseline_collect.exe --model test_data/test_models/ocr/ppocrv4_mobile/det_infer.onnx --image test_data/test_images/test_ocr.png --out tests/baselines --type pre
./build/bin/baseline_collect.exe --model test_data/test_models/ocr/ppocrv4_mobile/det_infer.onnx --image test_data/test_images/test_ocr.png --out tests/baselines --type raw
./build/bin/baseline_collect.exe --model test_data/test_models/ocr/ppocrv4_mobile/rec_infer.onnx --image test_data/test_images/test_ocr.png --out tests/baselines --type ocr_rec
./build/bin/baseline_collect.exe --model test_data/test_models/ocr/ppocrv4_mobile/cls_infer.onnx --image test_data/test_images/test_ocr.png --out tests/baselines --type ocr_cls
```
Expected: 全部文件生成，无报错

- [ ] **Step 2: 自洽验证**

```bash
./build/bin/test_modeldeploy.exe "[regression]"
```
Expected: 全部通过（收集器生成的基线 vs 对比器重新跑，无差异）

- [ ] **Step 3: 灵敏度验证（人为破坏）**

```powershell
# 备份
Copy-Item tests/baselines/yolo11n-obb_nms.onnx.obb.json C:\Users\aichao\AppData\Local\Temp\opencode\obb_baseline.bak
# 篡改第一个 angle
python -c "import json; d=json.load(open('tests/baselines/yolo11n-obb_nms.onnx.obb.json','r',encoding='utf-8')); d['results'][0]['angle']=88.0; json.dump(d,open('tests/baselines/yolo11n-obb_nms.onnx.obb.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)"
./build/bin/test_modeldeploy.exe "[regression]" 2>&1 | Select-String "angle mismatch|FAILED"
# 恢复
Copy-Item C:\Users\aichao\AppData\Local\Temp\opencode\obb_baseline.bak tests/baselines/yolo11n-obb_nms.onnx.obb.json -Force
```
Expected: 报出 `angle mismatch: base=88.000000 cur=<真实角度>` 且 FAILED；恢复后重新跑通过

- [ ] **Step 4: 提交基线文件**

```bash
git add tests/baselines/
git commit -m "test: 生成全部模型回归基线文件"
```

- [ ] **Step 5: 全量回归确认**

```bash
cd build
ctest --output-on-failure
```
Expected: 所有测试通过（含新增 [regression]）

---

## Self-Review

**Spec coverage:**
- 三类输出（pre/raw/后处理）→ Task 2/3 ✓
- 文本 JSON 存储、git 管理 → Task 2 生成 + Task 5 提交 ✓
- 浮动阈值 → Task 1 的 compare_* ✓
- ctest 自动跑 → Task 4 `[regression]` 集成 ✓
- 手动更新基线 → Task 2 collect + Task 5 说明 ✓
- OCR 纳入 → Task 2/3 的 ocr_det/ocr_rec/ocr_cls ✓
- 错误处理（缺失 skip、张量截断）→ Task 3 模板 + Task 1 serialize ✓

**待实现时确认的点（不阻塞计划）：**
- `MODELDEPLOY_CXX_EXPORT` 宏在测试 exe 内需为空定义（SDK 符号已导出到 dll）——Task 4 已加 `target_compile_definitions`
- face_det（scrfd）输出是 `KeyPointsResult`（含关键点），用 pose 序列化路径（字段相同）——compare_pose 可复用
- ocr_rec 需要 dict 文件路径（ppocrv4_dict.txt），Task 2 需处理
- `model.get_preprocessor().run` 的返回签名需在实现时对照具体模型（det/obb 有 LetterBox 记录，cls 无）——Task 3 模板按模型微调
