//
// Created by aichao on 2025/2/20.
//

#include "core/md_log.h"
#include "vision/common/processors/yolo_preproc.cuh"
#include "vision/detection/preprocessor.h"
#include "vision/common/processors/yolo_preproc.h"

namespace modeldeploy::vision::detection {
    UltralyticsPreprocessor::UltralyticsPreprocessor() {
        size_ = {640, 640};
        padding_value_ = {114.0, 114.0, 114.0};
    }


    bool UltralyticsPreprocessor::preprocess(ImageData* image, Tensor* output,
                                             LetterBoxRecord* letter_box_record) const {
        if (use_cuda_preproc_) {
#ifdef WITH_GPU
            return yolo_preprocess_cuda(image, output, size_, padding_value_, letter_box_record);
#else
            MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, rollback to cpu" << std::ebndl;
#endif
        }
        return yolo_preprocess_cpu(image, output, size_, padding_value_, letter_box_record);
    }

    bool UltralyticsPreprocessor::run(
        std::vector<ImageData>* images, std::vector<Tensor>* outputs,
        std::vector<LetterBoxRecord>* letter_box_records) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        letter_box_records->resize(images->size());
        outputs->resize(1);
        // Concat all the preprocessed data to a batch tensor
        std::vector<Tensor> tensors(images->size());
        for (size_t i = 0; i < images->size(); ++i) {
            // 修改了数据，并生成一个tensor,并记录预处理的一些参数，便于在后处理中还原
            preprocess(&(*images)[i], &tensors[i], &(*letter_box_records)[i]);
        }
        if (tensors.size() == 1) {
            (*outputs)[0] = std::move(tensors[0]);
        }
        else {
            (*outputs)[0] = std::move(Tensor::concat(tensors, 0));
        }
        return true;
    }
}
