#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/sam/preprocessor.h"

namespace modeldeploy::vision::seg {
    FastSamPreprocessor::FastSamPreprocessor() {
        size_ = {1024, 1024};
        padding_value_ = {114.0, 114.0, 114.0};
    }

    bool FastSamPreprocessor::preprocess(const ImageData& image, Tensor* output,
                                         LetterBoxRecord* letter_box_record) const {
        return backend_->yolo_preprocess(image, output, size_, padding_value_[0], letter_box_record);
    }

    bool FastSamPreprocessor::run(
        const std::vector<ImageData>& images, std::vector<Tensor>* outputs,
        std::vector<LetterBoxRecord>* letter_box_records) const {
        if (images.empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        letter_box_records->resize(images.size());
        outputs->resize(1);
        if (images.size() == 1) {
            return preprocess(images[0], &(*outputs)[0], &(*letter_box_records)[0]);
        }
        if (!backend_->yolo_preprocess_batch(images, &(*outputs)[0], size_, padding_value_[0],
                                             letter_box_records)) {
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::seg
