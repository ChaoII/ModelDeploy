#include "vision/tools/smoother.h"
#include <algorithm>
namespace modeldeploy::vision::tool {
Detections DetectionSmoother::update(const Detections& in) {
    Detections out = in;
    if (in.tracker_id.size() == in.boxes.size()) {
        if (state_.size() != in.boxes.size()) state_ = in.boxes;
        for (size_t i = 0; i < in.boxes.size(); ++i) {
            out.boxes[i].x = (float)(alpha_ * in.boxes[i].x + (1 - alpha_) * state_[i].x);
            out.boxes[i].y = (float)(alpha_ * in.boxes[i].y + (1 - alpha_) * state_[i].y);
        }
        state_ = out.boxes;
    } else {
        if (state_.empty()) return out;
        const size_t n = std::min(state_.size(), in.boxes.size());
        for (size_t i = 0; i < n; ++i) {
            out.boxes[i].x = (float)(alpha_ * in.boxes[i].x + (1 - alpha_) * state_[i].x);
            out.boxes[i].y = (float)(alpha_ * in.boxes[i].y + (1 - alpha_) * state_[i].y);
        }
        state_.resize(n);
        for (size_t i = 0; i < n; ++i) state_[i] = out.boxes[i];
    }
    return out;
}
} // namespace modeldeploy::vision::tool
