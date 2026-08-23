#pragma once
#include <utility>
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/tools/detections.h"

namespace modeldeploy::vision::tool {
class MODELDEPLOY_CXX_EXPORT LineZone {
public:
    LineZone(Point2f start, Point2f end) : a_(start), b_(end) {}
    void reset() { count_ = 0; last_in_ = false; has_last_ = false; }
    int trigger_count() const { return count_; }
    bool in_side() const { return has_last_ ? last_in_ : false; }
    bool trigger(const Point2f& p);
private:
    Point2f a_, b_;
    int count_{0};
    bool last_in_{false};
    bool has_last_{false};
};
class MODELDEPLOY_CXX_EXPORT PolygonZone {
public:
    PolygonZone() = default;
    explicit PolygonZone(std::vector<Point2f> points) : points_(std::move(points)) {}
    void reset() { count_ = 0; }
    bool contains(Point2f p) const;
    int current_count() const { return count_; }
    void update(const std::vector<Point2f>& pts);
    const std::vector<Point2f>& points() const { return points_; }
private:
    std::vector<Point2f> points_;
    int count_{0};
};
MODELDEPLOY_CXX_EXPORT void filter_by_zone(Detections& d, const PolygonZone& zone,
                                           const std::vector<int32_t>* keep_classes = nullptr,
                                           float score_threshold = 0.0f);
} // namespace modeldeploy::vision::tool
