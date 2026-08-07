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
            case DataType::FP64: return "FP64";
            case DataType::INT32: return "INT32";
            case DataType::INT64: return "INT64";
            case DataType::UINT8: return "UINT8";
            case DataType::INT8: return "INT8";
            default: return "UNKNOWN";
        }
    }

    template <typename... Args>
    static std::string fmt(const char* fmt, Args... args) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), fmt, args...);
        return buf;
    }

    static bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

    json serialize_tensor(const Tensor& t) {
        json j;
        j["name"] = t.get_name();
        j["shape"] = t.shape();
        j["dtype"] = dtype_to_str(t.dtype());
        const size_t n = t.size();
        j["numel"] = n;
        if (n == 0) {
            j["values"] = json::array();
            j["stats"] = { {"min", 0}, {"max", 0}, {"mean", 0} };
            return j;
        }
        const bool float_dtype = (t.dtype() == DataType::FP32 || t.dtype() == DataType::FP64);
        if (!float_dtype) {
            j["stats"] = { {"min", 0}, {"max", 0}, {"mean", 0} };
            return j;
        }
        const float* d = static_cast<const float*>(t.data());
        std::vector<float> vals;
        vals.reserve(std::min(n, MAX_TENSOR_ELEMS));
        for (size_t i = 0; i < n && i < MAX_TENSOR_ELEMS; ++i) {
            float v = d[i];
            vals.push_back(std::roundf(v * 1e6f) / 1e6f);
        }
        float mn = d[0], mx = d[0], sum = 0;
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
                    if (b["keypoints"][k].size() > 2)
                        check_coord("kp_z", b["keypoints"][k][2], rs[i].keypoints[k].z, &diffs);
                }
            }
        }
        return diffs;
    }

    std::vector<std::string> compare_cls(const json& base, const ClassifyResult& r) {
        std::vector<std::string> diffs;
        if (base["label_ids"].size() != r.label_ids.size())
            return {"label count mismatch"};
        if (base["scores"].size() != r.scores.size())
            return {"score count mismatch"};
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
        if (base["numel"].get<size_t>() != n)
            return {"tensor numel mismatch"};
        if (n == 0)
            return diffs;
        const bool float_dtype = (cur.dtype() == DataType::FP32 || cur.dtype() == DataType::FP64);
        if (!float_dtype)
            return {"tensor dtype not supported for value comparison"};
        const float* d = static_cast<const float*>(cur.data());
        // 统计比较
        float mn = d[0], mx = d[0], sum = 0;
        for (size_t i = 0; i < n; ++i) { if (d[i] < mn) mn = d[i]; if (d[i] > mx) mx = d[i]; sum += d[i]; }
        if (std::fabs((float)base["stats"]["min"] - mn) > TENSOR_TOL) diffs.push_back("tensor min mismatch");
        if (std::fabs((float)base["stats"]["max"] - mx) > TENSOR_TOL) diffs.push_back("tensor max mismatch");
        if (std::fabs((float)base["stats"]["mean"] - sum / n) > TENSOR_TOL) diffs.push_back("tensor mean mismatch");
        // 抽样比较
        size_t m = std::min(n, MAX_TENSOR_ELEMS);
        if (!base.contains("values") || base["values"].size() != m)
            return {"tensor values array mismatch"};
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
