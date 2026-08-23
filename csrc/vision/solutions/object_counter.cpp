#include "vision/solutions/object_counter.h"
#include <algorithm>
namespace modeldeploy::vision::solution {
void ObjectCounter::set_line(Point2f a, Point2f b) {
    line_pts_ = {a, b};
    line_zone_.clear();
    has_line_ = true;
    stats_.line_in = stats_.line_out = 0;
}
void ObjectCounter::set_region(const std::vector<Point2f>& pts) {
    region_ = tool::PolygonZone(pts);
    has_region_ = true;
    region_.reset();
}
void ObjectCounter::set_classes(const std::vector<int32_t>& cls) { classes_ = cls; }
void ObjectCounter::update(const std::vector<tracking::TrackResult>& tracks) {
    for (const auto& t : tracks) {
        const Point2f c(t.box.x + t.box.width * 0.5f, t.box.y + t.box.height * 0.5f);
        if (has_line_) {
            auto itz = line_zone_.find(t.track_id);
            if (itz == line_zone_.end()) {
                itz = line_zone_.emplace(t.track_id, tool::LineZone(line_pts_.first, line_pts_.second)).first;
            }
            if (itz->second.trigger(c)) {
                if (itz->second.in_side()) ++stats_.line_in; else ++stats_.line_out;
            }
        }
        if (has_region_) {
            auto it = last_centroid_.find(t.track_id);
            const bool prev_in = it != last_centroid_.end() && region_.contains(it->second);
            const bool now_in = region_.contains(c);
            if (now_in && !prev_in) ++region_count_;
        }
        if (classes_.empty() || std::find(classes_.begin(), classes_.end(), t.label_id) != classes_.end())
            stats_.class_count[t.label_id]++;
        last_centroid_[t.track_id] = c;
    }
}
void ObjectCounter::reset() {
    region_.reset(); line_zone_.clear(); last_centroid_.clear();
    stats_ = CounterStats{}; region_count_ = 0;
}
} // namespace modeldeploy::vision::solution
