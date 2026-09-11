//
// ServingServer 每请求结果参数：对结果 JSON 做通用后过滤。
//
// 支持参数（in["params"]）：
//   threshold  置信度下限（<0 或缺省=不过滤）—— 过滤结果项；分类/OCR 按各自分数。
//   top_k      分类结果取前 K（>0 生效）。
//   max_det    目标数上限（>0 生效）。
//
// 说明：阈值作用于结果 JSON 与前端叠加/列表；服务端渲染图由模型默认阈值生成。
//
#pragma once

#include <algorithm>
#include <nlohmann/json.hpp>

namespace modeldeploy::serving {
    namespace detail {

        inline double item_score(const nlohmann::json& it) {
            if (it.is_object()) {
                if (it.contains("score") && it["score"].is_number())
                    return it["score"].get<double>();
                if (it.contains("box_score") && it["box_score"].is_number())
                    return it["box_score"].get<double>();
            }
            return -1.0;
        }

        // 按保留索引重建并行数组（保证 boxes/text/rec_scores/cls_* 对齐）。
        inline void rebuild_parallel(nlohmann::json& obj, const char* key,
                                     const std::vector<size_t>& keep) {
            if (!obj.contains(key) || !obj[key].is_array()) return;
            nlohmann::json out = nlohmann::json::array();
            const auto& src = obj[key];
            for (size_t i : keep)
                if (i < src.size()) out.push_back(src[i]);
            obj[key] = std::move(out);
        }

        inline void apply_result_params(nlohmann::json& results, const nlohmann::json& params) {
            if (!params.is_object()) return;
            const double thr = params.value("threshold", -1.0);
            const int top_k = params.value("top_k", 0);
            const int max_det = params.value("max_det", 0);

            if (results.is_array()) {
                if (thr >= 0.0) {
                    results.erase(
                        std::remove_if(results.begin(), results.end(),
                                       [&](const nlohmann::json& it) {
                                           const double s = item_score(it);
                                           return s >= 0.0 && s < thr;
                                       }),
                        results.end());
                }
                if (max_det > 0 && static_cast<int>(results.size()) > max_det)
                    results.erase(results.begin() + max_det, results.end());
                return;
            }
            if (!results.is_object()) return;

            // 分类：scores / label_ids 成对
            if (results.contains("scores") && results["scores"].is_array()) {
                const bool has_ids =
                    results.contains("label_ids") && results["label_ids"].is_array();
                std::vector<size_t> keep;
                const auto& sc = results["scores"];
                for (size_t i = 0; i < sc.size(); ++i) {
                    const double s = sc[i].is_number() ? sc[i].get<double>() : -1.0;
                    if (thr >= 0.0 && s >= 0.0 && s < thr) continue;
                    keep.push_back(i);
                    if (top_k > 0 && static_cast<int>(keep.size()) >= top_k) break;
                }
                rebuild_parallel(results, "scores", keep);
                if (has_ids) rebuild_parallel(results, "label_ids", keep);
                if (results.contains("feature") && results["feature"].is_array()
                    && results["feature"].size() != keep.size()) {
                    // feature 为整段向量，不随类别过滤（保持原样）
                }
                return;
            }

            // OCR：boxes/text/rec_scores/cls_scores/cls_labels 并行数组，按 rec_scores 过滤行
            if (results.contains("text") && results["text"].is_array()
                && results.contains("boxes") && results["boxes"].is_array()) {
                const bool has_sc =
                    results.contains("rec_scores") && results["rec_scores"].is_array();
                const auto& sc = results["rec_scores"];
                const auto& tx = results["text"];
                std::vector<size_t> keep;
                for (size_t i = 0; i < tx.size(); ++i) {
                    const double s =
                        (has_sc && i < sc.size() && sc[i].is_number()) ? sc[i].get<double>() : -1.0;
                    if (thr >= 0.0 && s >= 0.0 && s < thr) continue;
                    if (max_det > 0 && static_cast<int>(keep.size()) >= max_det) break;
                    keep.push_back(i);
                }
                rebuild_parallel(results, "boxes", keep);
                rebuild_parallel(results, "text", keep);
                if (has_sc) rebuild_parallel(results, "rec_scores", keep);
                rebuild_parallel(results, "cls_scores", keep);
                rebuild_parallel(results, "cls_labels", keep);
                return;
            }
        }

    }  // namespace detail
}  // namespace modeldeploy::serving
