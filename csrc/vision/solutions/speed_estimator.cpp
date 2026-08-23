#include "vision/solutions/speed_estimator.h"
#include <cmath>
namespace modeldeploy::vision::solution {
void SpeedEstimator::update(const std::vector<tracking::TrackResult>& tracks, double timestamp_ms) {
    px_per_s_.clear();
    for (const auto& t : tracks) {
        const Point2f c(t.box.x + t.box.width * 0.5f, t.box.y + t.box.height * 0.5f);
        auto it = last_.find(t.track_id);
        if (it != last_.end()) {
            const double dt = timestamp_ms - it->second.second;
            const float dx = c.x - it->second.first.x;
            const float dy = c.y - it->second.first.y;
            const float dist = std::sqrt(dx * dx + dy * dy);
            if (dt > 0.0) px_per_s_[t.track_id] = dist / (float)dt * 1000.0f;
        }
        last_[t.track_id] = {c, timestamp_ms};
    }
}
std::map<int,float> SpeedEstimator::speeds_m_s() const {
    std::map<int,float> out;
    for (const auto& kv : px_per_s_) out[kv.first] = kv.second * mpp_;
    return out;
}
void SpeedEstimator::reset() { last_.clear(); px_per_s_.clear(); }
} // namespace modeldeploy::vision::solution
