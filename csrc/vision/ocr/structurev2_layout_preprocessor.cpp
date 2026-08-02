//
// Created by aichao on 2025/3/21.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/normalize_and_permute.h"
#include "vision/ocr/structurev2_layout_preprocessor.h"


namespace modeldeploy::vision::ocr {
    StructureV2LayoutPreprocessor::StructureV2LayoutPreprocessor() {
    }

    std::array<int, 4> StructureV2LayoutPreprocessor::get_layout_image_info(ImageData* image) {
        cv::Mat img;
        image->to_mat(img);
        if (static_shape_infer_) {
            return {
                img.cols, img.rows, layout_image_shape_[2],
                layout_image_shape_[1]
            };
        }
        MD_LOG_ERROR << "not support dynamic shape inference now!" << std::endl;
        return {
            img.cols, img.rows, layout_image_shape_[2], layout_image_shape_[1]
        };
    }


    bool StructureV2LayoutPreprocessor::run(std::vector<ImageData>* image_batch,
                                            std::vector<Tensor>* outputs) {
        batch_layout_img_info_.clear();
        batch_layout_img_info_.resize(image_batch->size());
        std::vector<Tensor> tensors;
        tensors.reserve(image_batch->size());
        for (size_t i = 0; i < image_batch->size(); ++i) {
            ImageData* image = &image_batch->at(i);
            batch_layout_img_info_[i] = get_layout_image_info(image);
            ImageData resized;
            if (!backend_->resize(*image, &resized, batch_layout_img_info_[i][2], batch_layout_img_info_[i][3])) return false;
            ImageData normed;
            if (!backend_->normalize(resized, &normed, mean_, std_, is_scale_, false)) return false;
            Tensor t;
            if (!backend_->hwc2chw(normed, &t)) return false;
            t.expand_dim(0);
            tensors.emplace_back(std::move(t));
        }
        outputs->resize(1);
        (*outputs)[0] = Tensor::concat(tensors, 0);
        return true;
    }
} // namespace modeldeploy::vision::ocr
