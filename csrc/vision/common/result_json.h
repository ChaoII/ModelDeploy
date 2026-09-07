// ModelDeploy: 推理结果 → JSON 序列化。
// 将 vision 各 Result 结构导出为 nlohmann::json，供上层协议 / Python / CAPI 直接交换。
// 独立头文件，按需 include；result.h 本身不依赖 nlohmann。
#pragma once

#include <nlohmann/json.hpp>

#include "vision/common/result.h"

namespace modeldeploy::vision {

    inline nlohmann::json to_json(const Rect2f& r) {
        return {{"x", r.x}, {"y", r.y}, {"width", r.width}, {"height", r.height}};
    }

    inline nlohmann::json to_json(const Point3f& p) {
        return {{"x", p.x}, {"y", p.y}, {"z", p.z}};
    }

    inline nlohmann::json to_json(const RotatedRect& r) {
        return {{"xc", r.xc}, {"yc", r.yc}, {"width", r.width}, {"height", r.height}, {"angle", r.angle}};
    }

    inline nlohmann::json to_json(const Mask& m) {
        nlohmann::json j;
        j["shape"] = m.shape;
        j["data"] = nlohmann::json::array();
        for (size_t i = 0; i < m.buffer.size(); ++i) j["data"].push_back(m.buffer[i]);
        return j;
    }

    inline nlohmann::json to_json(const ClassifyResult& r) {
        return {{"label_ids", r.label_ids}, {"scores", r.scores}, {"feature", r.feature}};
    }

    inline nlohmann::json to_json(const DetectionResult& r) {
        return {{"box", nlohmann::json(to_json(r.box))},
                {"label_id", r.label_id},
                {"score", r.score}};
    }

    inline nlohmann::json to_json(const InstanceSegResult& r) {
        return {{"box", nlohmann::json(to_json(r.box))},
                {"mask", to_json(r.mask)},
                {"label_id", r.label_id},
                {"score", r.score}};
    }

    inline nlohmann::json to_json(const SemSegResult& r) {
        nlohmann::json j;
        j["shape"] = r.shape;
        j["num_classes"] = r.num_classes;
        j["labels"] = nlohmann::json::array();
        for (size_t i = 0; i < r.labels.size(); ++i) j["labels"].push_back(r.labels[i]);
        return j;
    }

    inline nlohmann::json to_json(const DepthResult& r) {
        return {{"shape", r.shape}, {"depth", r.depth}};
    }

    inline nlohmann::json to_json(const ObbResult& r) {
        return {{"rotated_box", nlohmann::json(to_json(r.rotated_box))},
                {"label_id", r.label_id},
                {"score", r.score}};
    }

    inline nlohmann::json to_json(const KeyPointsResult& r) {
        nlohmann::json kps = nlohmann::json::array();
        for (const auto& p : r.keypoints) kps.push_back(to_json(p));
        return {{"box", nlohmann::json(to_json(r.box))},
                {"keypoints", kps},
                {"label_id", r.label_id},
                {"score", r.score}};
    }

    inline nlohmann::json to_json(const OCRResult& r) {
        return {{"boxes", r.boxes},          {"text", r.text},
                {"rec_scores", r.rec_scores},{"cls_scores", r.cls_scores},
                {"cls_labels", r.cls_labels},{"table_boxes", r.table_boxes},
                {"table_structure", r.table_structure},
                {"table_html", r.table_html}};
    }

    inline nlohmann::json to_json(const FaceRecognitionResult& r) {
        return {{"embedding", r.embedding}};
    }

    inline nlohmann::json to_json(const ReIdResult& r) {
        return {{"embedding", r.embedding}};
    }

    inline nlohmann::json to_json(const LprResult& r) {
        nlohmann::json kps = nlohmann::json::array();
        for (const auto& p : r.keypoints) kps.push_back(to_json(p));
        return {{"box", nlohmann::json(to_json(r.box))},
                {"keypoints", kps},
                {"label_id", r.label_id},
                {"score", r.score},
                {"car_plate_str", r.car_plate_str},
                {"car_plate_color", r.car_plate_color}};
    }

    inline nlohmann::json to_json(const AttributeResult& r) {
        return {{"box", nlohmann::json(to_json(r.box))},
                {"box_label_id", r.box_label_id},
                {"box_score", r.box_score},
                {"attr_scores", r.attr_scores}};
    }

    // 通用：任意单对象可 to_json 的 T，其 std::vector<T> 自动展开为 JSON 数组。
    // 注意必须与各 to_json(T) 同命名空间（modeldeploy::vision），依赖 ADL 找到逐元素 to_json。
    template <typename T>
    inline nlohmann::json to_json(const std::vector<T>& rs) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& r : rs) arr.push_back(to_json(r));
        return arr;
    }

}  // namespace modeldeploy::vision
