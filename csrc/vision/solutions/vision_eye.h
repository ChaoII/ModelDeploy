#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT VisionEye : public SolutionBase {
public:
    explicit VisionEye(float eye_level_y = 0.0f) : eye_level_(eye_level_y) {}
    static Point2f map_to_eye(Point2f centroid, float eye_level_y);
    void add(Point2f centroid);
    const std::vector<Point2f>& eyes() const { return eyes_; }
    void reset() override { eyes_.clear(); }
private:
    float eye_level_;
    std::vector<Point2f> eyes_;
};
} // namespace modeldeploy::vision::solution
