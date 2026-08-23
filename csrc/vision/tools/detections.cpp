#include "vision/tools/detections.h"
#include <algorithm>
#include <cmath>

namespace modeldeploy::vision::tool {
float iou(const Rect2f& a, const Rect2f& b) {
    const float ax2 = a.x + a.width, ay2 = a.y + a.height;
    const float bx2 = b.x + b.width, by2 = b.y + b.height;
    const float ix = std::max(0.0f, std::min(ax2, bx2) - std::max(a.x, b.x));
    const float iy = std::max(0.0f, std::min(ay2, by2) - std::max(a.y, b.y));
    const float inter = ix * iy;
    const float uni = a.width * a.height + b.width * b.height - inter;
    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}
void nms(Detections& d, float iou_threshold) {
    const size_t n = d.size();
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b){ return d.confidence[a] > d.confidence[b]; });
    std::vector<bool> keep(n, true);
    for (size_t i = 0; i < n; ++i) {
        if (!keep[order[i]]) continue;
        for (size_t j = i + 1; j < n; ++j)
            if (keep[order[j]] && iou(d.boxes[order[i]], d.boxes[order[j]]) > iou_threshold)
                keep[order[j]] = false;
    }
    Detections out; out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!keep[order[i]]) continue;
        out.boxes.push_back(d.boxes[order[i]]);
        out.class_id.push_back(d.class_id[order[i]]);
        out.confidence.push_back(d.confidence[order[i]]);
        if (!d.masks.empty()) out.masks.push_back(d.masks[order[i]]);
        if (!d.tracker_id.empty()) out.tracker_id.push_back(d.tracker_id[order[i]]);
    }
    d = std::move(out);
}
void filter_by_class(Detections& d, const std::vector<int32_t>& keep) {
    Detections out; out.reserve(d.size());
    for (size_t i = 0; i < d.size(); ++i) {
        if (std::find(keep.begin(), keep.end(), d.class_id[i]) == keep.end()) continue;
        out.boxes.push_back(d.boxes[i]); out.class_id.push_back(d.class_id[i]);
        if (!d.confidence.empty()) out.confidence.push_back(d.confidence[i]);
        if (!d.masks.empty()) out.masks.push_back(d.masks[i]);
        if (!d.tracker_id.empty()) out.tracker_id.push_back(d.tracker_id[i]);
    }
    d = std::move(out);
}
Detections from_track(const std::vector<tracking::TrackResult>& t) {
    Detections d; d.reserve(t.size());
    for (const auto& r : t) {
        d.boxes.push_back(r.box); d.class_id.push_back(r.label_id);
        d.confidence.push_back(r.score); d.tracker_id.push_back(r.track_id);
    }
    return d;
}
Detections from_detections(const std::vector<tracking::Detection>& t) {
    Detections d; d.reserve(t.size());
    for (const auto& r : t) {
        d.boxes.push_back(r.box); d.class_id.push_back(r.label_id);
        d.confidence.push_back(r.score);
    }
    return d;
}
} // namespace modeldeploy::vision::tool
