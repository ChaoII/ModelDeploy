//
// insightface buffalo_l w600k_r50 后处理：输出向量直接作为 embedding。
//
#pragma once

#include <vector>
#include "core/tensor.h"
#include "core/md_decl.h"

namespace modeldeploy::vision::face {

    class MODELDEPLOY_CXX_EXPORT InsightFaceRecPostprocessor {
    public:
        bool run(const std::vector<Tensor>& infer_results, std::vector<float>* embedding);
    };

} // namespace modeldeploy::vision::face
