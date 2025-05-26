//
// Created by aichao on 2025/2/24.
//
#include "postprocessor.h"
#include "csrc/vision/utils.h"


namespace modeldeploy::vision::classification {
    YOLOv5ClsPostprocessor::YOLOv5ClsPostprocessor() {
        topk_ = 1;
    }

    bool YOLOv5ClsPostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<ClassifyResult>* results,
        const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info) {
        const int64_t batch = tensors[0].shape()[0];
        const Tensor& infer_result = tensors[0];
        Tensor infer_result_softmax = infer_result.softmax(1);
        results->resize(batch);

        for (size_t bs = 0; bs < batch; ++bs) {
            (*results)[bs].clear();
            // output (1,1000) score class_num 1000
            int64_t num_classes = infer_result_softmax.shape()[1];
            const float* infer_result_buffer =
                static_cast<const float*>(infer_result_softmax.data()) + bs * infer_result_softmax.shape()[1];
            topk_ = std::min(static_cast<int>(num_classes), topk_);
            (*results)[bs].label_ids =
                utils::top_k_indices(infer_result_buffer, num_classes, topk_);
            (*results)[bs].scores.resize(topk_);
            for (int i = 0; i < topk_; ++i) {
                (*results)[bs].scores[i] = *(infer_result_buffer + (*results)[bs].label_ids[i]);
            }

            if ((*results)[bs].label_ids.empty()) {
                return true;
            }
        }
        return true;
    }
} // namespace modeldeploy::vision::classification
