#include "vision/solutions/region_counter.h"
#include <algorithm>
namespace modeldeploy::vision::solution {
void RegionCounter::add_region(const std::string& name, const std::vector<Point2f>& polygon) {
    if (polygon.size() < 3) return;
    for (auto& r : regions_) {
        if (r.name == name) { r.zone = tool::PolygonZone(polygon); return; }
    }
    regions_.push_back(Region{name, tool::PolygonZone(polygon)});
    counts_[name] = 0;
}
void RegionCounter::set_classes(const std::vector<int32_t>& cls) { classes_ = cls; has_classes_ = !cls.empty(); }
void RegionCounter::update(const std::vector<tracking::TrackResult>& tracks) {
    for (auto& kv : counts_) kv.second = 0;
    for (const auto& t : tracks) {
        if (has_classes_ && std::find(classes_.begin(), classes_.end(), t.label_id) == classes_.end()) continue;
        const Point2f c(t.box.x + t.box.width * 0.5f, t.box.y + t.box.height * 0.5f);
        for (const auto& r : regions_) if (r.zone.contains(c)) ++counts_[r.name];
    }
}
void RegionCounter::reset() { regions_.clear(); counts_.clear(); classes_.clear(); has_classes_ = false; }
} // namespace modeldeploy::vision::solution
