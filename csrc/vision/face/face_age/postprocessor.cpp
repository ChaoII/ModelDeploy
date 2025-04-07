//
// Created by aichao on 2025/3/24.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/utils.h"
#include "csrc/core/md_type.h"
#include "csrc/vision/face/face_age/postprocessor.h"

namespace modeldeploy::vision::face {
    bool SeetaFaceAgePostprocessor::run(std::vector<MDTensor>& infer_result, std::vector<int>* ages) {
        if (infer_result[0].dtype != MDDataType::Type::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        if (infer_result.size() != 1) {
            MD_LOG_ERROR << "The default number of output tensor "
                "must be 1 according to insightface." << std::endl;
        }

        const size_t batch = infer_result[0].shape[0];
        ages->resize(batch);

        for (size_t bs = 0; bs < batch; ++bs) {
            MDTensor& age_tensor = infer_result.at(bs);
            if (age_tensor.shape[0] != 1) {
                MD_LOG_ERROR << "Only support batch = 1 now." << std::endl;
            }
            if (age_tensor.dtype != MDDataType::Type::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const int ele_size = age_tensor.total();
            auto* age_tensor_ptr = static_cast<float*>(age_tensor.data());
            std::vector embedding(age_tensor_ptr, age_tensor_ptr + ele_size);
            float age_f = 0.0f;
            for (int i = 0; i < ele_size; i++) {
                age_f += embedding[i] * static_cast<float>(i);
            }
            (*ages)[bs] = static_cast<int>(age_f + 0.5f);
        }
        return true;
    }
}
