#pragma once
#include <utility>
#include <vector>
#include "core/md_decl.h"
#include "vision/tracking/base_tracker.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT Heatmap : public SolutionBase {
public:
    Heatmap() = default;
    void set_size(int w, int h);
    void update(const std::vector<tracking::TrackResult>& tracks, int frame_w, int frame_h);
    const std::vector<float>& heat() const { return heat_; }
    std::pair<int,int> peak() const;
    float heat_at(int x, int y) const;
    void reset() override;
private:
    int w_{0}, h_{0};
    std::vector<float> heat_;
};
} // namespace modeldeploy::vision::solution
