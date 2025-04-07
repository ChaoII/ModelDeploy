//
// Created by aichao on 2025/3/24.
//

#include "csrc/core/md_log.h"
#include "csrc/core/md_type.h"
#include "csrc/vision/face/face_gender/postprocessor.h"

namespace modeldeploy::vision::face {
    bool SeetaFaceGenderPostprocessor::run(std::vector<MDTensor>& infer_result, std::vector<int>* genders) {
        if (infer_result[0].dtype != MDDataType::Type::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        if (infer_result.size() != 1) {
            MD_LOG_ERROR << "The default number of output tensor must be 1." << std::endl;
        }

        const size_t batch = infer_result[0].shape[0];
        genders->resize(batch);
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
}
