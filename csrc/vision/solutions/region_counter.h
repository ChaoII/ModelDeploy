#pragma once
#include <map>
#include <string>
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/tracking/base_tracker.h"
#include "vision/tools/zone.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT RegionCounter : public SolutionBase {
public:
    void add_region(const std::string& name, const std::vector<Point2f>& polygon);
    void set_classes(const std::vector<int32_t>& cls);
    void update(const std::vector<tracking::TrackResult>& tracks);
    std::map<std::string, int> region_counts() const { return counts_; }
    size_t total_regions() const { return regions_.size(); }
    void reset() override;
private:
    struct Region { std::string name; tool::PolygonZone zone; };
    std::vector<Region> regions_;
    std::map<std::string, int> counts_;
    std::vector<int32_t> classes_;
    bool has_classes_{false};
};
} // namespace modeldeploy::vision::solution
