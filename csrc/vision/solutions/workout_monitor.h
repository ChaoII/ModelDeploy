#pragma once
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT WorkoutMonitor : public SolutionBase {
public:
    WorkoutMonitor(float min_deg = 70.0f, float max_deg = 160.0f) : min_(min_deg), max_(max_deg) {}
    static float angle(Point3f a, Point3f b, Point3f c);
    int reps() const { return reps_; }
    void update(float elbow_deg);
    void reset() override { reps_ = 0; down_ = false; }
private:
    float min_, max_;
    int reps_{0};
    bool down_{false};
};
} // namespace modeldeploy::vision::solution
