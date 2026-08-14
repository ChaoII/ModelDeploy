//
// insightface buffalo_l w600k_r50 后处理实现。
//
#include "vision/face/insightface/recognition/insightface_recognition_postprocessor.h"

namespace modeldeploy::vision::face {

    bool InsightFaceRecPostprocessor::run(const std::vector<Tensor>& infer_results,
                                          std::vector<float>* embedding) {
        if (infer_results.empty()) return false;
        const float* out = static_cast<const float*>(infer_results[0].data());
        const size_t dim = infer_results[0].size();
        embedding->resize(dim);
        for (size_t i = 0; i < dim; ++i) (*embedding)[i] = out[i];
        return true;
    }

} // namespace modeldeploy::vision::face
