//
// Created by aichao on 2025/3/21.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/ocr/structurev2_layout_preprocessor.h"


namespace modeldeploy::vision::ocr {
    StructureV2LayoutPreprocessor::StructureV2LayoutPreprocessor() {
        // default width(608) and height(900)
        resize_op_ =
            std::make_shared<Resize>(layout_image_shape_[2], layout_image_shape_[1]);
        normalize_permute_op_ = std::make_shared<NormalizeAndPermute>(
            std::vector({0.485f, 0.456f, 0.406f}),
            std::vector({0.229f, 0.224f, 0.225f}), true);
    }

    std::array<int, 4> StructureV2LayoutPreprocessor::get_layout_image_info(cv::Mat* img) {
        if (static_shape_infer_) {
            return {
                img->cols, img->rows, layout_image_shape_[2],
                layout_image_shape_[1]
            };
        }
        MD_LOG_ERROR << "not support dynamic shape inference now!" << std::endl;
        return {
            img->cols, img->rows, layout_image_shape_[2], layout_image_shape_[1]
        };
    }

    bool StructureV2LayoutPreprocessor::resize_layout_image(cv::Mat* img,
                                                            const int resize_w,
                                                            const int resize_h) const {
        resize_op_->set_width_and_height(resize_w, resize_h);
        (*resize_op_)(img);
        return true;
    }

    bool StructureV2LayoutPreprocessor::run(std::vector<cv::Mat>* image_batch,
                                            std::vector<Tensor>* outputs) {
        batch_layout_img_info_.clear();
        batch_layout_img_info_.resize(image_batch->size());
        for (size_t i = 0; i < image_batch->size(); ++i) {
            cv::Mat* mat = &image_batch->at(i);
            batch_layout_img_info_[i] = get_layout_image_info(mat);
            resize_layout_image(mat, batch_layout_img_info_[i][2],
                                batch_layout_img_info_[i][3]);
            (*normalize_permute_op_)(mat);
        }

        outputs->resize(1);
        utils::mats_to_tensor(*image_batch, &(*outputs)[0]);
        return true;
    }
} // namespace modeldeploy::vision::ocr
