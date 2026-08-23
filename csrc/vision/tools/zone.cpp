#include "vision/tools/zone.h"
#include <algorithm>

namespace modeldeploy::vision::tool {
static bool side_in(const Point2f& a, const Point2f& b, const Point2f& p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) < 0.0f;
}
bool LineZone::trigger(const Point2f& p) {
    const bool in = side_in(a_, b_, p);
    bool crossed = false;
    if (has_last_ && last_in_ != in) crossed = true; // 两侧翻转计一次
    if (crossed) ++count_;
    last_in_ = in;
    has_last_ = true;
    return crossed;
}
bool PolygonZone::contains(Point2f p) const {
    bool inside = false;
    const int n = static_cast<int>(points_.size());
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const Point2f& a = points_[i];
        const Point2f& b = points_[j];
        if ((a.y > p.y) != (b.y > p.y) &&
            p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)
            inside = !inside;
    }
    return inside;
}
void PolygonZone::update(const std::vector<Point2f>& pts) {
    for (const auto& p : pts) if (contains(p)) ++count_;
}
void filter_by_zone(Detections& d, const PolygonZone& zone,
                    const std::vector<int32_t>* keep_classes, float score_threshold) {
    Detections out; out.reserve(d.size());
    for (size_t i = 0; i < d.size(); ++i) {
        if (!d.confidence.empty() && d.confidence[i] < score_threshold) continue;
        if (keep_classes && std::find(keep_classes->begin(), keep_classes->end(), d.class_id[i]) == keep_classes->end()) continue;
        const Rect2f& b = d.boxes[i];
        if (!zone.contains(Point2f(b.x + b.width * 0.5f, b.y + b.height * 0.5f))) continue;
        out.boxes.push_back(d.boxes[i]); out.class_id.push_back(d.class_id[i]);
        if (!d.confidence.empty()) out.confidence.push_back(d.confidence[i]);
        if (!d.masks.empty()) out.masks.push_back(d.masks[i]);
        if (!d.tracker_id.empty()) out.tracker_id.push_back(d.tracker_id[i]);
    }
    d = std::move(out);
}
} // namespace modeldeploy::vision::tool
