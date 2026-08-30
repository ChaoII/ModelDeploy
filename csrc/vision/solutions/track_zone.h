#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/tracking/base_tracker.h"
#include "vision/tools/zone.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT TrackZone : public SolutionBase {
public:
    void set_region(const std::vector<Point2f>& polygon);
    void set_classes(const std::vector<int32_t>& cls);
    void update(const std::vector<tracking::TrackResult>& tracks);
    std::vector<tracking::TrackResult> inside_tracks() const { return inside_; }
    int inside_count() const { return static_cast<int>(inside_.size()); }
    void reset() override;
private:
    tool::PolygonZone region_;
    bool has_region_{false};
    std::vector<int32_t> classes_;
    bool has_classes_{false};
    std::vector<tracking::TrackResult> inside_;
};
} // namespace modeldeploy::vision::solution
