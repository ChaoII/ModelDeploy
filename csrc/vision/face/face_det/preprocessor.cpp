//
// Created by aichao on 2025/2/20.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/face/face_det/preprocessor.h"
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
        return backend_->scrfd_preprocess(*image, output, size_,
                                          static_cast<float>(padding_value_[0]),
                                          letter_box_record);
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
        if (images->size() == 1) {
            // 单图直接写到持久 outputs[0]（复用已有 buffer）
            preprocess(&(*images)[0], &(*outputs)[0], &(*letter_box_records)[0]);
        }
        else {
            // 多图整批一次 kernel（CPU/CUDA/Sophgo 各自实现），避免 N 次 launch + concat
            backend_->scrfd_preprocess_batch(*images, &(*outputs)[0], size_,
                                             static_cast<float>(padding_value_[0]),
                                             letter_box_records);
        }
        return true;
    }
}
