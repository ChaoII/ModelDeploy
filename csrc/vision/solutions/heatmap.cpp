#include "vision/solutions/heatmap.h"
#include <algorithm>
namespace modeldeploy::vision::solution {
void Heatmap::set_size(int w, int h) { w_ = w; h_ = h; heat_.assign((size_t)w * h, 0.0f); }
void Heatmap::update(const std::vector<tracking::TrackResult>& tracks, int frame_w, int frame_h) {
    if (heat_.empty() || w_ <= 0 || h_ <= 0) return;
    const float sx = (float)w_ / (float)std::max(1, frame_w);
    const float sy = (float)h_ / (float)std::max(1, frame_h);
    for (const auto& t : tracks) {
        const float cx = t.box.x + t.box.width * 0.5f;
        const float cy = t.box.y + t.box.height * 0.5f;
        int x = std::max(0, std::min(w_ - 1, (int)(cx * sx)));
        int y = std::max(0, std::min(h_ - 1, (int)(cy * sy)));
        heat_[(size_t)y * w_ + x] += 1.0f;
    }
}
std::pair<int,int> Heatmap::peak() const {
    if (heat_.empty()) return {0, 0};
    auto it = std::max_element(heat_.begin(), heat_.end());
    if (*it <= 0.0f) return {0, 0};
    const size_t idx = (size_t)(it - heat_.begin());
    return {(int)(idx % w_), (int)(idx / w_)};
}
float Heatmap::heat_at(int x, int y) const {
    if (heat_.empty() || x < 0 || y < 0 || x >= w_ || y >= h_) return 0.0f;
    return heat_[(size_t)y * w_ + x];
}
void Heatmap::reset() { std::fill(heat_.begin(), heat_.end(), 0.0f); }
} // namespace modeldeploy::vision::solution
