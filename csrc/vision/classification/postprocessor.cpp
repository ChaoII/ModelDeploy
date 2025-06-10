//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision/utils.h"
#include "csrc/vision/classification/postprocessor.h"

namespace modeldeploy::vision::classification {
    UltralyticsClsPostprocessor::UltralyticsClsPostprocessor() {
        top_k_ = 1;
    }

    bool UltralyticsClsPostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<ClassifyResult>* results) const {
        const int64_t batch = tensors[0].shape()[0];
        const Tensor& infer_result = tensors[0];
        // 注意yolov8-cls在模型中已经做过softmax了。
        // infer_result = infer_result.softmax(1);
        results->resize(batch);

        for (size_t bs = 0; bs < batch; ++bs) {
            (*results)[bs].clear();
            // output (1,1000) score class_num 1000
            const int64_t num_classes = infer_result.shape()[1];
            const float* infer_result_buffer =
                static_cast<const float*>(infer_result.data()) + bs * infer_result.shape()[1];
            const auto top_k = std::min(static_cast<int>(num_classes), top_k_);
            (*results)[bs].label_ids =
                utils::top_k_indices(infer_result_buffer, num_classes, top_k);
            (*results)[bs].scores.resize(top_k);
            for (int i = 0; i < top_k; ++i) {
                (*results)[bs].scores[i] = *(infer_result_buffer + (*results)[bs].label_ids[i]);
            }

            if ((*results)[bs].label_ids.empty()) {
                return true;
            }
        }
        return true;
    }
} // namespace modeldeploy::vision::classification
