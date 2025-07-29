//
// Created by aichao on 2025/2/21.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/ocr/rec_preprocessor.h"
#include "vision/common/processors/pad.h"
#include "vision/common/processors/cast.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/normalize_and_permute.h"

namespace modeldeploy::vision::ocr {
    RecognizerPreprocessor::RecognizerPreprocessor() {
    }

    bool RecognizerPreprocessor::run(const std::vector<ImageData>* images,
                                     std::vector<Tensor>* outputs,
                                     const size_t start_index, const size_t end_index,
                                     const std::vector<int>& indices) const {
        if (images->empty() || end_index <= start_index ||
            end_index > images->size()) {
            MD_LOG_ERROR << "images->size() or index error. Correct is: 0 <= start_index < "
                "end_index <= images->size()" << std::endl;;
            return false;
        }

        std::vector<ImageData> mats(end_index - start_index);
        for (size_t i = start_index; i < end_index; ++i) {
            size_t real_index = i;
            if (!indices.empty()) {
                real_index = indices[i];
            }
            mats[i - start_index] = images->at(real_index);
        }
        return apply(&mats, outputs);
    }

    bool RecognizerPreprocessor::apply(const std::vector<ImageData>* image_batch,
                                       std::vector<Tensor>* outputs) const {
        const int img_h = rec_image_shape_[1];
        const int img_w = rec_image_shape_[2];
        float max_wh_ratio = static_cast<float>(img_w) * 1.0f / static_cast<float>(img_h);
        for (const auto& image : *image_batch) {
            float ori_wh_ratio = static_cast<float>(image.width()) * 1.0f / static_cast<float>(image.height());
            max_wh_ratio = std::max(max_wh_ratio, ori_wh_ratio);
        }
        std::vector<cv::Mat> _images;
        for (auto& image : *image_batch) {
            cv::Mat mat;
            image.to_mat(&mat);
            const int img_h_ = rec_image_shape_[1];
            int img_w_ = rec_image_shape_[2];
            if (!static_shape_infer_) {
                img_w_ = static_cast<int>(static_cast<float>(img_h_) * max_wh_ratio);
                const float ratio = static_cast<float>(mat.cols) / static_cast<float>(mat.rows);
                int resize_w;
                if (ceilf(static_cast<float>(img_h_) * ratio) > static_cast<float>(img_w_)) {
                    resize_w = img_w_;
                }
                else {
                    resize_w = static_cast<int>(ceilf(static_cast<float>(img_h_) * ratio));
                }
                Resize::apply(&mat, resize_w, img_h_);
                Pad::apply(&mat, 0, 0, 0, img_w_ - mat.cols, pad_value_);
            }
            else {
                if (mat.cols >= img_w_) {
                    // Resize W to 320
                    Resize::apply(&mat, img_w_, img_h_);
                }
                else {
                    Resize::apply(&mat, mat.cols, img_h_);
                    // Pad to 320
                    Pad::apply(&mat, 0, 0, 0, img_w_ - mat.cols, pad_value_);
                }
            }
            NormalizeAndPermute::apply(&mat, mean_, std_, is_scale_);
            _images.push_back(mat);
        }
        // Only have 1 output Tensor.
        outputs->resize(1);
        utils::mats_to_tensor(_images, &(*outputs)[0]);
        return true;
    }
}
