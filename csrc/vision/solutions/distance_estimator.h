#pragma once
#include <utility>
#include <vector>
#include "core/md_decl.h"
#include "vision/tracking/base_tracker.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT DistanceEstimator : public SolutionBase {
public:
    void set_meter_per_pixel(float m) { mpp_ = m; }
    std::vector<std::pair<std::pair<int,int>, float>> pair_distances_px(const std::vector<tracking::TrackResult>& tracks);
    std::vector<std::pair<std::pair<int,int>, float>> pair_distances_m(const std::vector<tracking::TrackResult>& tracks);
    void reset() override {}
private:
    float mpp_{0.01f};
};
} // namespace modeldeploy::vision::solution
