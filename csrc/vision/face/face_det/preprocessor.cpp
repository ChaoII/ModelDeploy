//
// Created by aichao on 2025/2/20.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/face/face_det/preprocessor.h"
#include <vision/common/processors/cast.h>
#include <vision/common/processors/color_space_convert.h>
#include <vision/common/processors/convert.h>
#include <vision/common/processors/hwc2chw.h>
#ifdef WITH_GPU
#include "vision/face/face_det/scrfd_preproc.cuh"
#endif
#include "vision/face/face_det/scrfd_preproc.h"

namespace modeldeploy::vision::face {
    ScrfdPreprocessor::ScrfdPreprocessor() {
        size_ = {640, 640};
        padding_value_ = {0.0, 0.0, 0.0};
        is_mini_pad_ = false;
        is_no_pad_ = false;
        is_scale_up_ = true;
        stride_ = 32;
    }


    bool ScrfdPreprocessor::preprocess(ImageData* image, Tensor* output, LetterBoxRecord* letter_box_record) const {
        if (use_cuda_preproc_) {
#ifdef WITH_GPU
            return scrfd_preprocess_cuda(*image, output, size_, static_cast<float>(padding_value_[0]), letter_box_record);
#else
            MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, fallback to cpu" << std::endl;
#endif
        }
        return scrfd_preprocess_cpu(*image, output, size_, static_cast<float>(padding_value_[0]), letter_box_record);
    }

    bool ScrfdPreprocessor::run(
        std::vector<ImageData>* images, std::vector<Tensor>* outputs,
        std::vector<LetterBoxRecord>* letter_box_records) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        letter_box_records->resize(1);
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
