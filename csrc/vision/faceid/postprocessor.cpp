//
// Created by aichao on 2025/3/24.
//

#include "csrc/vision/faceid/postprocessor.h"
#include "csrc/vision/utils.h"
#include "csrc/core/md_type.h"

namespace modeldeploy::vision::faceid {

  AdaFacePostprocessor::AdaFacePostprocessor() {
    l2_normalize_ = false;
  }

  bool AdaFacePostprocessor::Run(std::vector<MDTensor>& infer_result,
                                 std::vector<FaceRecognitionResult>* results) {
    if (infer_result[0].dtype != MDDataType::Type::FP32) {
      std::cerr << "Only support post process with float32 data." << std::endl;
      return false;
    }
    if(infer_result.size() != 1){
      std::cerr  << "The default number of output tensor "
        "must be 1 according to insightface." << std::endl;
    }
    int batch = infer_result[0].shape[0];
    results->resize(batch);
    for (size_t bs = 0; bs < batch; ++bs) {
      MDTensor& embedding_tensor = infer_result.at(bs);
      if((embedding_tensor.shape[0] != 1)){
        std::cerr<<"Only support batch = 1 now."<<std::endl;}
      if (embedding_tensor.dtype != MDDataType::Type::FP32) {
        std::cerr << "Only support post process with float32 data." << std::endl;
        return false;
      }
      (*results)[bs].Clear();
      (*results)[bs].Resize(embedding_tensor.total());

      // Copy the raw embedding vector directly without L2 normalize
      // post process. Let the user decide whether to normalize or not.
      // Will call utils::L2Normlize() method to perform L2
      // normalize if l2_normalize was set as 'true'.
      std::memcpy((*results)[bs].embedding.data(),
                  embedding_tensor.data(),
                  embedding_tensor.total_bytes());
      if (l2_normalize_) {
        auto norm_embedding = utils::l2_normalize((*results)[bs].embedding);
        std::memcpy((*results)[bs].embedding.data(),
                    norm_embedding.data(),
                    embedding_tensor.total_bytes());
      }
    }
    return true;
  }
}
