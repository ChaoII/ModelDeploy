#pragma once
#include <map>
#include <utility>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/tracking/base_tracker.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT SpeedEstimator : public SolutionBase {
public:
    void set_meter_per_pixel(float m) { mpp_ = m; }
    void update(const std::vector<tracking::TrackResult>& tracks, double timestamp_ms);
    std::map<int,float> speeds_px_per_s() const { return px_per_s_; }
    std::map<int,float> speeds_m_s() const;
    void reset() override;
private:
    float mpp_{0.01f};
    std::map<int, std::pair<Point2f, double>> last_;
    std::map<int,float> px_per_s_;
};
} // namespace modeldeploy::vision::solution
