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
