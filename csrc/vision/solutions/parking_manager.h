#pragma once
#include <algorithm>
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/tracking/base_tracker.h"
#include "vision/tools/zone.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT ParkingManager : public SolutionBase {
public:
    void set_slots(const std::vector<std::vector<Point2f>>& slots);
    void update(const std::vector<tracking::TrackResult>& tracks);
    std::vector<bool> occupancy() const { return occupied_; }
    void reset() override { std::fill(occupied_.begin(), occupied_.end(), false); }
private:
    std::vector<tool::PolygonZone> slots_;
    std::vector<bool> occupied_;
};
} // namespace modeldeploy::vision::solution
