//
// Created by aichao on 2025/3/24.
//


#include "csrc/core/md_log.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/face/face_rec/postprocessor.h"


namespace modeldeploy::vision::face {
    bool SeetaFaceIDPostprocessor::run(std::vector<Tensor>& infer_result,
                                       std::vector<FaceRecognitionResult>* results) {
        if (infer_result[0].dtype() != DataType::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        if (infer_result.size() != 1) {
            MD_LOG_ERROR << "The default number of output tensor "
                "must be 1 according to insightface." << std::endl;
        }
        const size_t batch = infer_result[0].shape()[0];
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            Tensor& embedding_tensor = infer_result.at(bs);
            if (embedding_tensor.shape()[0] != 1) {
                MD_LOG_ERROR << "Only support batch = 1 now." << std::endl;
            }
            if (embedding_tensor.dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            (*results)[bs].embedding.clear();
            (*results)[bs].embedding.resize(embedding_tensor.size());

            std::memcpy((*results)[bs].embedding.data(),
                        embedding_tensor.data(),
                        embedding_tensor.byte_size());
            // 这里是seeta face的特殊处理，计算平方根再计算l2标准化
            (*results)[bs].embedding = utils::compute_sqrt((*results)[bs].embedding);
            auto norm_embedding = utils::l2_normalize((*results)[bs].embedding);
            std::memcpy((*results)[bs].embedding.data(),
                        norm_embedding.data(),
                        embedding_tensor.byte_size());
        }
        return true;
    }
}
