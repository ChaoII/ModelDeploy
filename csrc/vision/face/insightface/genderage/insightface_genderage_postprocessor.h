//
// insightface buffalo_l genderage 后处理：输出 [1,3] -> gender(argmax[:2]) + age(round(pred[2]*100))。
//
#pragma once

#include <vector>
#include "core/tensor.h"
#include "core/md_decl.h"

namespace modeldeploy::vision::face {

    struct GenderAgeResult {
        int gender = -1; // 0=male, 1=female
        int age = -1;
    };

    class MODELDEPLOY_CXX_EXPORT InsightFaceGenderAgePostprocessor {
    public:
        bool run(const std::vector<Tensor>& infer_results, GenderAgeResult* result) const;
    };

} // namespace modeldeploy::vision::face
