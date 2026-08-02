//
// Created by aichao on 2025/2/20.
//

#include "core/md_log.h"
#include "vision/detection/preprocessor.h"

namespace modeldeploy::vision::detection {
    UltralyticsPreprocessor::UltralyticsPreprocessor() {
        size_ = {640, 640};
        padding_value_ = {114.0, 114.0, 114.0};
    }


    bool UltralyticsPreprocessor::preprocess(const ImageData& image, Tensor* output,
                                             LetterBoxRecord* letter_box_record) const {
        return backend_->yolo_preprocess(image, output, size_, padding_value_[0], letter_box_record);
    }

    bool UltralyticsPreprocessor::run(const uint8_t* src_y,
                                      const uint8_t* src_uv,
                                      const std::vector<int>& src_size,
                                      const int step_y,
                                      const int step_uv,
                                      Tensor* output,
                                      LetterBoxRecord* letter_box_record) const {
        return backend_->yolo_preprocess_nv12(src_y, src_uv, src_size,
                                              step_y, step_uv, output, size_,
                                              padding_value_[0], letter_box_record);
    }


    bool UltralyticsPreprocessor::run(const std::vector<ImageData>& images, std::vector<Tensor>* outputs,
                                      std::vector<LetterBoxRecord>* letter_box_records) const {
        if (images.empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        letter_box_records->resize(images.size());
        outputs->resize(1);
        if (images.size() == 1) {
            // 单图：直接写进持久 outputs[0]，其 allocate 跨帧复用显存 buffer
            return preprocess(images[0], &(*outputs)[0], &(*letter_box_records)[0]);
        }
        // 多图：逐图预处理后 concat
        std::vector<Tensor> tensors(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
            // 修改了数据，并生成一个tensor,并记录预处理的一些参数，便于在后处理中还原
            preprocess(images[i], &tensors[i], &(*letter_box_records)[i]);
        }
        (*outputs)[0] = std::move(Tensor::concat(tensors, 0));
        return true;
    }
}
