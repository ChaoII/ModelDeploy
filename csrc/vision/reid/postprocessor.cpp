//
// Created for standalone pedestrian Re-ID (OSNet) postprocessing.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/reid/postprocessor.h"

namespace modeldeploy::vision::reid {
    bool ReIDPostprocessor::run(const std::vector<Tensor>& inputs,
                                std::vector<std::vector<ReIdResult>>* results) {
        if (inputs.empty()) {
            MD_LOG_ERROR << "The input tensors should not be empty." << std::endl;
            return false;
        }
        const Tensor& in = inputs[0];
        if (in.dtype() != DataType::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        const auto& shape = in.shape();
        if (shape.empty()) {
            MD_LOG_ERROR << "The output tensor shape should not be empty." << std::endl;
            return false;
        }
        const size_t batch = static_cast<size_t>(shape[0]);
        const size_t total = in.size();  // 元素总数
        if (batch == 0 || total % batch != 0) {
            MD_LOG_ERROR << "Invalid output tensor shape for Re-ID." << std::endl;
            return false;
        }
        const size_t dim = total / batch;  // 每样本 embedding 维度（OSNet 为 512）

        const float* data = static_cast<const float*>(in.data());
        results->clear();
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            const float* row = data + bs * dim;
            std::vector<float> embedding(row, row + dim);
            results->at(bs).push_back(ReIdResult{utils::l2_normalize(embedding)});
        }
        return true;
    }
} // namespace modeldeploy::vision::reid
