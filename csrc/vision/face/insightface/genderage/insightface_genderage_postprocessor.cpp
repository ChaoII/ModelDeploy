//
// insightface buffalo_l genderage 后处理实现。
// python: gender = argmax(pred[:2]); age = int(round(pred[2]*100))
//
#include "vision/face/insightface/genderage/insightface_genderage_postprocessor.h"
#include <cmath>

namespace modeldeploy::vision::face {

    bool InsightFaceGenderAgePostprocessor::run(const std::vector<Tensor>& infer_results,
                                                GenderAgeResult* result) const {
        if (infer_results.empty() || result == nullptr) return false;
        const float* pred = static_cast<const float*>(infer_results[0].data());
        const int total = static_cast<int>(infer_results[0].size());
        if (total < 3) return false;
        result->gender = (pred[0] >= pred[1]) ? 0 : 1;
        result->age = static_cast<int>(std::lround(pred[2] * 100.0f));
        return true;
    }

} // namespace modeldeploy::vision::face
