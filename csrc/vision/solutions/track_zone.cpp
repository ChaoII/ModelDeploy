#include "vision/solutions/track_zone.h"
#include <algorithm>
namespace modeldeploy::vision::solution {
void TrackZone::set_region(const std::vector<Point2f>& polygon) {
    if (polygon.size() < 3) { has_region_ = false; return; }
    region_ = tool::PolygonZone(polygon); has_region_ = true;
}
void TrackZone::set_classes(const std::vector<int32_t>& cls) { classes_ = cls; has_classes_ = !cls.empty(); }
void TrackZone::update(const std::vector<tracking::TrackResult>& tracks) {
    inside_.clear();
    if (!has_region_) return;
    for (const auto& t : tracks) {
        if (has_classes_ && std::find(classes_.begin(), classes_.end(), t.label_id) == classes_.end()) continue;
        const Point2f c(t.box.x + t.box.width * 0.5f, t.box.y + t.box.height * 0.5f);
        if (region_.contains(c)) inside_.push_back(t);
    }
}
void TrackZone::reset() { region_ = tool::PolygonZone{}; has_region_ = false; classes_.clear(); has_classes_ = false; inside_.clear(); }
} // namespace modeldeploy::vision::solution
