//
// Created by aichao on 2025/2/21.
//

#include "cls_postprocessor.h"
#include "./utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    bool single_batch_postprocessor(const float* out_data, const size_t& length,
                                  int* cls_label, float* cls_score) {
        *cls_label = std::distance(&out_data[0],
                                   std::max_element(&out_data[0], &out_data[length]));
        *cls_score = static_cast<float>(*std::max_element(&out_data[0], &out_data[length]));
        return true;
    }

    bool ClassifierPostprocessor::run(const std::vector<MDTensor>& tensors,
                                      std::vector<int32_t>* cls_labels,
                                      std::vector<float>* cls_scores) {
        const size_t total_size = tensors[0].shape[0];
        return run(tensors, cls_labels, cls_scores, 0, total_size);
    }

    bool ClassifierPostprocessor::run(const std::vector<MDTensor>& tensors,
                                      std::vector<int32_t>* cls_labels,
                                      std::vector<float>* cls_scores,
                                      size_t start_index, size_t total_size) {
        // Classifier have only 1 output tensor.
        const MDTensor& tensor = tensors[0];

        // For Classifier, the output tensor shape = [batch,2]
        const size_t batch = tensor.shape[0];
        const size_t length = accumulate(tensor.shape.begin() + 1, tensor.shape.end(), 1,
                                   std::multiplies<int>());
        if (batch <= 0) {
            std::cerr << "The infer outputTensor.shape[0] <=0, wrong infer result."<< std::endl;
            return false;
        }
        if (total_size <= 0) {
            std::cerr << "start_index or total_size error. Correct is: 0 <= start_index "
                "< total_size"<< std::endl;
            return false;
        }
        if ((start_index + batch) > total_size) {
            std::cerr << "start_index or total_size error. Correct is: start_index + "
                "batch(outputTensor.shape[0]) <= total_size"
                << std::endl;
            return false;
        }
        cls_labels->resize(total_size);
        cls_scores->resize(total_size);
        const float* tensor_data = static_cast<const float*>(tensor.data());
        for (int i_batch = 0; i_batch < batch; ++i_batch) {
            single_batch_postprocessor(tensor_data + i_batch * length,
                                     length,
                                     &cls_labels->at(i_batch + start_index),
                                     &cls_scores->at(i_batch + start_index));
        }

        return true;
    }
}
