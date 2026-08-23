#include "vision/solutions/distance_estimator.h"
#include <cmath>
namespace modeldeploy::vision::solution {
static Point2f centroid(const tracking::TrackResult& t) {
    return Point2f(t.box.x + t.box.width * 0.5f, t.box.y + t.box.height * 0.5f);
}
std::vector<std::pair<std::pair<int,int>, float>> DistanceEstimator::pair_distances_px(
        const std::vector<tracking::TrackResult>& tracks) {
    std::vector<std::pair<std::pair<int,int>, float>> out;
    std::vector<std::pair<int,Point2f>> pts;
    for (const auto& t : tracks) pts.emplace_back(t.track_id, centroid(t));
    for (size_t i = 0; i < pts.size(); ++i)
        for (size_t j = i + 1; j < pts.size(); ++j) {
            const float dx = pts[i].second.x - pts[j].second.x;
            const float dy = pts[i].second.y - pts[j].second.y;
            out.emplace_back(std::make_pair(pts[i].first, pts[j].first), std::sqrt(dx*dx + dy*dy));
        }
    return out;
}
std::vector<std::pair<std::pair<int,int>, float>> DistanceEstimator::pair_distances_m(
        const std::vector<tracking::TrackResult>& tracks) {
    auto px = pair_distances_px(tracks);
    for (auto& e : px) e.second *= mpp_;
    return px;
}
} // namespace modeldeploy::vision::solution
