#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/common/result.h"
#include "vision/tracking/base_tracker.h"

namespace modeldeploy::vision::tool {
struct MODELDEPLOY_CXX_EXPORT Detections {
    std::vector<Rect2f> boxes;
    std::vector<int32_t> class_id;
    std::vector<float> confidence;
    std::vector<Mask> masks;
    std::vector<int32_t> tracker_id;
    [[nodiscard]] size_t size() const { return boxes.size(); }
    void reserve(size_t n) { boxes.reserve(n); class_id.reserve(n); confidence.reserve(n); masks.reserve(n); tracker_id.reserve(n); }
};
MODELDEPLOY_CXX_EXPORT float iou(const Rect2f& a, const Rect2f& b);
MODELDEPLOY_CXX_EXPORT void nms(Detections& d, float iou_threshold);
MODELDEPLOY_CXX_EXPORT void filter_by_class(Detections& d, const std::vector<int32_t>& keep_classes);
MODELDEPLOY_CXX_EXPORT Detections from_track(const std::vector<tracking::TrackResult>& t);
MODELDEPLOY_CXX_EXPORT Detections from_detections(const std::vector<tracking::Detection>& t);
} // namespace modeldeploy::vision::tool
