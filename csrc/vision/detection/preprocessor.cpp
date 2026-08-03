//
// Created by aichao on 2025/2/20.
//

#include "core/md_log.h"
#include "vision/detection/preprocessor.h"
#include "vision/utils.h"

namespace modeldeploy::vision::detection {
    UltralyticsPreprocessor::UltralyticsPreprocessor() {
        size_ = {640, 640};
        padding_value_ = {114.0, 114.0, 114.0};
    }


    bool UltralyticsPreprocessor::preprocess(const ImageData& image, Tensor* output,
                                             LetterBoxRecord* letter_box_record) const {
        if (!normalize_) {
            // 无归一化路径：letterbox 后保持 [0,255]，alpha=1（部分导出模型期望原始像素）
            *letter_box_record = utils::cal_letter_box_param(
                {image.width(), image.height()}, size_);
            float ox, oy, sx, sy;
            utils::letter_box_to_fused_params(*letter_box_record, &ox, &oy, &sx, &sy);
            const std::vector<float> alpha = {1.0f, 1.0f, 1.0f};
            const std::vector<float> beta = {0.0f, 0.0f, 0.0f};
            return backend_->fused_preprocess(image, output, size_,
                                              ox, oy, sx, sy, alpha, beta,
                                              true, padding_value_[0]);
        }
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
        // 多图：一次融合 batch kernel（GPU 3D grid），避免 N 次 launch + concat
        if (!backend_->yolo_preprocess_batch(images, &(*outputs)[0], size_, padding_value_[0],
                                             letter_box_records)) {
            return false;
        }
        return true;
    }
}
