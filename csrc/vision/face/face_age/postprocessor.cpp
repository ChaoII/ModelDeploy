//
// Created by aichao on 2025/3/24.
//

#include "core/md_log.h"
#include "vision/face/face_age/postprocessor.h"

namespace modeldeploy::vision::face {
    bool SeetaFaceAgePostprocessor::run(const std::vector<Tensor>& infer_result, std::vector<int>* ages) const {
        if (infer_result[0].dtype() != DataType::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        // (-1,82,1,1)
        const size_t batch = infer_result[0].shape()[0];
        ages->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            const auto dim1 = infer_result[0].shape()[1]; //82
            const auto dim2 = infer_result[0].shape()[2]; //1
            const auto dim3 = infer_result[0].shape()[3]; //1
            const float* age_tensor_ptr = static_cast<const float*>(infer_result[0].data()) + bs * dim1 * dim2 * dim3;
            const auto ele_size = dim1 * dim2 * dim3;
            std::vector embedding(age_tensor_ptr, age_tensor_ptr + ele_size);
            float age_f = 0.0f;
            for (int i = 0; i < ele_size; i++) {
                age_f += embedding[i] * static_cast<float>(i);
            }
            (*ages)[bs] = static_cast<int>(age_f + 0.5f);
        }
        return true;
    }

    bool SeetaFaceAgePostprocessor::run_dist(
        const std::vector<Tensor>& infer_result, std::vector<std::vector<float>>* dists) const {
        if (infer_result.empty() || dists == nullptr) return false;
        if (infer_result[0].dtype() != DataType::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        const auto& shape = infer_result[0].shape();
        if (shape.size() < 2) return false;
        // 输出 [N, C, ...]（如 [N,88,1,1]）；每张脸的年龄分布 = 该 batch 的 C 个元素
        const size_t batch = static_cast<size_t>(shape[0]);
        size_t ele_size = 1;
        for (size_t d = 1; d < shape.size(); ++d) ele_size *= static_cast<size_t>(shape[d]);
        const float* src = static_cast<const float*>(infer_result[0].data());
        dists->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            const float* p = src + bs * ele_size;
            (*dists)[bs].assign(p, p + ele_size);
        }
        return true;
    }
}
