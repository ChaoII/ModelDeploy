//
// Created by aichao on 2025/2/24.
//

#include "vision/utils.h"
#include "vision/classification/postprocessor.h"

#include <numeric>

namespace modeldeploy::vision::classification {
    ClassificationPostprocessor::ClassificationPostprocessor() {
        top_k_ = 1;
    }

    bool ClassificationPostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<ClassifyResult>* results) const {
        const int64_t batch = tensors[0].shape()[0];
        const Tensor& infer_result = tensors[0];
        // 注意cls在模型中已经做过softmax了。
        // infer_result = infer_result.softmax(1);
        // 对于多标签分类，score为每个类别的概率，label_id为类别的索引。
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            ClassifyResult r;
            // output (1,1000) score class_num 1000
            const int64_t num_classes = infer_result.shape()[1];
            const float* infer_result_buffer =
                static_cast<const float*>(infer_result.data()) + bs * infer_result.shape()[1];
            if (multi_label_) {
                r.label_ids.resize(num_classes);
                std::iota(r.label_ids.begin(),
                          r.label_ids.end(),0);
                r.scores = std::vector<float>(infer_result_buffer, infer_result_buffer + num_classes);
            }
            else {
                const auto top_k = std::min(static_cast<int>(num_classes), top_k_);
                r.label_ids =
                    utils::top_k_indices(infer_result_buffer, static_cast<int>(num_classes), top_k);
                r.scores.resize(top_k);
                for (int i = 0; i < top_k; ++i) {
                    r.scores[i] = *(infer_result_buffer + r.label_ids[i]);
                }
            }
            if (r.label_ids.empty()) {
                return false;
            }
            (*results)[bs] = std::move(r);
        }
        return true;
    }
} // namespace modeldeploy::vision::classification
