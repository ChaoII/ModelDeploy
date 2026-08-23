#pragma once
#include <map>
#include <utility>
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/tracking/base_tracker.h"
#include "vision/tools/zone.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
struct MODELDEPLOY_CXX_EXPORT CounterStats { int line_in{0}; int line_out{0}; std::map<int, int> class_count; };
class MODELDEPLOY_CXX_EXPORT ObjectCounter : public SolutionBase {
public:
    ObjectCounter() = default;
    void set_line(Point2f a, Point2f b);
    void set_region(const std::vector<Point2f>& pts);
    void set_classes(const std::vector<int32_t>& cls);
    void update(const std::vector<tracking::TrackResult>& tracks);
    CounterStats stats() const { return stats_; }
    int region_count() const { return region_count_; }
    void reset() override;
private:
    tool::PolygonZone region_;
    bool has_line_{false};
    bool has_region_{false};
    std::vector<int32_t> classes_;
    std::pair<Point2f, Point2f> line_pts_;
    std::map<int, Point2f> last_centroid_;
    std::map<int, tool::LineZone> line_zone_;
    CounterStats stats_;
    int region_count_{0};
};
} // namespace modeldeploy::vision::solution
