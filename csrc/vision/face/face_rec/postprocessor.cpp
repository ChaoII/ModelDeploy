//
// Created by aichao on 2025/3/24.
//


#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/face/face_rec/postprocessor.h"


namespace modeldeploy::vision::face {
    bool SeetaFaceIDPostprocessor::run(const std::vector<Tensor>& tensors,
                                       std::vector<FaceRecognitionResult>* results) {
        if (tensors[0].dtype() != DataType::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        // (-1,1024,1,1) -> (-1,1,1,1024)
        Tensor tensor_transpose = tensors[0].transpose({0, 2, 3, 1}).to_tensor();
        const int batch = tensor_transpose.shape()[0];
        const size_t dim3 = tensor_transpose.shape()[3];
        results->clear();
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            float* data = static_cast<float*>(tensor_transpose.data()) + bs * dim3;
            auto embedding = std::vector(data, data + dim3);
            // 这里是seeta face的特殊处理，计算平方根再计算l2标准化
            embedding = utils::compute_sqrt(embedding);
            const auto norm_embedding = utils::l2_normalize(embedding);
            (*results)[bs].embedding = norm_embedding;
        }
        return true;
    }
}
