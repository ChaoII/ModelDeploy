#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/tools/detections.h"
namespace modeldeploy::vision::tool {
class MODELDEPLOY_CXX_EXPORT DetectionSmoother {
public:
    explicit DetectionSmoother(double alpha = 0.5) : alpha_(alpha) {}
    void reset() { state_.clear(); }
    Detections update(const Detections& in);
private:
    double alpha_;
    std::vector<Rect2f> state_;
};
} // namespace modeldeploy::vision::tool
