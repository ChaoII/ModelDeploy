//
// Created by aichao on 2026/8/13.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/sem/preprocessor.h"

namespace modeldeploy::vision::detection {
    UltralyticsSemPreprocessor::UltralyticsSemPreprocessor() {
        size_ = {640, 640};
        padding_value_ = {114.0, 114.0, 114.0};
    }

    bool UltralyticsSemPreprocessor::preprocess(const ImageData& image, Tensor* output,
                                                LetterBoxRecord* letter_box_record) const {
        return backend_->yolo_preprocess(image, output, size_, padding_value_[0], letter_box_record);
    }

    bool UltralyticsSemPreprocessor::run(
        const std::vector<ImageData>& images, std::vector<Tensor>* outputs,
        std::vector<LetterBoxRecord>* letter_box_records) const {
        if (images.empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        letter_box_records->resize(images.size());
        outputs->resize(1);
        std::vector<Tensor> tensors(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
            preprocess(images[i], &tensors[i], &(*letter_box_records)[i]);
        }
        if (tensors.size() == 1) {
            (*outputs)[0] = std::move(tensors[0]);
        } else {
            (*outputs)[0] = std::move(Tensor::concat(tensors, 0));
        }
        return true;
    }
}
