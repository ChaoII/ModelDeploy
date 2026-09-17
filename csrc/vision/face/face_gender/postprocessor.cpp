//
// Created by aichao on 2025/3/24.
//

#include "core/md_log.h"
#include "vision/face/face_gender/postprocessor.h"

namespace modeldeploy::vision::face {
    bool SeetaFaceGenderPostprocessor::run(const std::vector<Tensor>& infer_result, std::vector<int>* genders) const {
        if (infer_result[0].dtype() != DataType::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        const size_t batch = infer_result[0].shape()[0];
        genders->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            const auto dim1 = infer_result[0].shape()[1];
            const auto dim2 = infer_result[0].shape()[2];
            const auto dim3 = infer_result[0].shape()[2];
            const auto* age_tensor_ptr = static_cast<const float*>(infer_result[0].data()) + bs * dim1 * dim2 * dim3;
            const auto ele_size = dim1 * dim2 * dim3;
            std::vector embedding(age_tensor_ptr, age_tensor_ptr + ele_size);
            if (embedding[0] > embedding[1]) {
                // female
                (*genders)[bs] = 0;
            }
            else {
                // male
                (*genders)[bs] = 1;
            }
        }
        return true;
    }

    bool SeetaFaceGenderPostprocessor::run_probs(
        const std::vector<Tensor>& infer_result, std::vector<std::vector<float>>* probs) const {
        if (infer_result.empty() || probs == nullptr) return false;
        if (infer_result[0].dtype() != DataType::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        const auto& shape = infer_result[0].shape();
        if (shape.size() < 2) return false;
        // 输出 [N, C, ...]（如 [N,2,1,1]）；每张脸的类别概率 = 该 batch 的 C 个元素
        const size_t batch = static_cast<size_t>(shape[0]);
        size_t ele_size = 1;
        for (size_t d = 1; d < shape.size(); ++d) ele_size *= static_cast<size_t>(shape[d]);
        const float* src = static_cast<const float*>(infer_result[0].data());
        probs->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            const float* p = src + bs * ele_size;
            (*probs)[bs].assign(p, p + ele_size);
        }
        return true;
    }
}
