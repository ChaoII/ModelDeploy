//
// Created by aichao on 2025/2/24.
//
#include "postprocessor.h"
#include "csrc/function/softmax.h"
#include "csrc/vision/utils.h"


namespace modeldeploy {
namespace vision {
namespace classification {

YOLOv5ClsPostprocessor::YOLOv5ClsPostprocessor() {
  topk_ = 1;
}

bool YOLOv5ClsPostprocessor::Run(
    const std::vector<MDTensor> &tensors, std::vector<ClassifyResult> *results,
    const std::vector<std::map<std::string, std::array<float, 2>>> &ims_info) {
  int batch = tensors[0].shape[0];
  MDTensor infer_result = tensors[0];
  MDTensor infer_result_softmax;
  function::softmax(infer_result, &infer_result_softmax, 1);
  results->resize(batch);

  for (size_t bs = 0; bs < batch; ++bs) {
    (*results)[bs].Clear();
    // output (1,1000) score classnum 1000
    int num_classes = infer_result_softmax.shape[1];
    const float* infer_result_buffer =
        reinterpret_cast<const float*>(infer_result_softmax.data()) + bs * infer_result_softmax.shape[1];
    topk_ = std::min(num_classes, topk_);
    (*results)[bs].label_ids =
        top_k_indices(infer_result_buffer, num_classes, topk_);
    (*results)[bs].scores.resize(topk_);
    for (int i = 0; i < topk_; ++i) {
      (*results)[bs].scores[i] = *(infer_result_buffer + (*results)[bs].label_ids[i]);
    }

    if ((*results)[bs].label_ids.size() == 0) {
      return true;
    }
  }
  return true;
}

} // namespace classification
} // namespace vision
} // namespace fastdeploy
