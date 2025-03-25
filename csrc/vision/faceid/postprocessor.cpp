//
// Created by aichao on 2025/3/24.
//

#include "csrc/vision/faceid/postprocessor.h"
#include "csrc/vision/utils.h"
#include "csrc/core/md_type.h"

namespace modeldeploy::vision::faceid {
    bool AdaFacePostprocessor::Run(std::vector<MDTensor>& infer_result,
                                   std::vector<FaceRecognitionResult>* results) {
        if (infer_result[0].dtype != MDDataType::Type::FP32) {
            std::cerr << "Only support post process with float32 data." << std::endl;
            return false;
        }
        if (infer_result.size() != 1) {
            std::cerr << "The default number of output tensor "
                "must be 1 according to insightface." << std::endl;
        }
        const size_t batch = infer_result[0].shape[0];
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            MDTensor& embedding_tensor = infer_result.at(bs);
            if ((embedding_tensor.shape[0] != 1)) {
                std::cerr << "Only support batch = 1 now." << std::endl;
            }
            if (embedding_tensor.dtype != MDDataType::Type::FP32) {
                std::cerr << "Only support post process with float32 data." << std::endl;
                return false;
            }
            (*results)[bs].Clear();
            (*results)[bs].Resize(embedding_tensor.total());

            std::memcpy((*results)[bs].embedding.data(),
                        embedding_tensor.data(),
                        embedding_tensor.total_bytes());
            // 这里是seeta face的特殊处理，计算平方根再计算l2标准化
            (*results)[bs].embedding = utils::compute_sqrt((*results)[bs].embedding);
            auto norm_embedding = utils::l2_normalize((*results)[bs].embedding);
            std::memcpy((*results)[bs].embedding.data(),
                        norm_embedding.data(),
                        embedding_tensor.total_bytes());
        }
        return true;
    }
}
