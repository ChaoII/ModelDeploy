//
// Created by aichao on 2025/2/21.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/ocr/cls_preprocessor.h"

#include <vision/common/processors/normalize_and_permute.h>

#include "vision/common/processors/pad.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/normalize.h"
#include "vision/common/processors/hwc2chw.h"
#include "vision/ocr/utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    ClassifierPreprocessor::ClassifierPreprocessor() {
    }

    bool ClassifierPreprocessor::run(const std::vector<ImageData>* images,
                                     std::vector<Tensor>* outputs,
                                     const size_t start_index, const size_t end_index) {
        if (static_cast<int>(images->size()) == 0 || end_index <= start_index ||
            end_index > images->size()) {
            MD_LOG_ERROR << "images->size() or index error. Correct is: 0 <= start_index < "
                "end_index <= images->size()" << std::endl;
            return false;
        }
        std::vector<ImageData> mats(end_index - start_index);
        for (size_t i = start_index; i < end_index; ++i) {
            mats[i - start_index] = images->at(i);
        }
        return apply(&mats, outputs);
    }

    bool ClassifierPreprocessor::apply(std::vector<ImageData>* image_batch,
                                       std::vector<Tensor>* outputs) {
        std::vector<cv::Mat> _images;
        for (auto& image : *image_batch) {
            cv::Mat mat;
            image.to_mat(&mat);
            const int img_h = cls_image_shape_[1];
            const int img_w = cls_image_shape_[2];
            const float ratio = static_cast<float>(mat.cols) / static_cast<float>(mat.rows);
            int resize_w;
            if (ceilf(static_cast<float>(img_h) * ratio) > static_cast<float>(img_w))
                resize_w = img_w;
            else
                resize_w = static_cast<int>(ceilf(static_cast<float>(img_h) * ratio));
            Resize::apply(&mat, resize_w, img_h);
            Normalize::apply(&mat, mean_, std_, is_scale_);
            std::vector<float> value = {0, 0, 0};
            if (mat.cols < cls_image_shape_[2]) {
                Pad::apply(&mat, 0, 0, 0, cls_image_shape_[2] - mat.cols, value);
            }
            HWC2CHW::apply(&mat);
            _images.push_back(mat);
        }
        // Only have 1 output tensor.
        outputs->resize(1);
        // Get the NCHW tensor
        utils::mats_to_tensor(_images, &(*outputs)[0]);
        return true;
    }
}
