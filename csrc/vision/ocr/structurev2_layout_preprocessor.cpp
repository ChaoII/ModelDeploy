//
// Created by aichao on 2025/3/21.
//

#include "core/md_log.h"
#include "vision/ocr/utils/ocr_utils.h"
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
            const int dst_w = batch_layout_img_info_[i][2];
            const int dst_h = batch_layout_img_info_[i][3];
            const int src_w = image->width();
            const int src_h = image->height();
            const float scale_x = static_cast<float>(dst_w) / src_w;
            const float scale_y = static_cast<float>(dst_h) / src_h;
            std::vector<float> alpha(3), beta(3);
            for (int c = 0; c < 3; ++c) {
                alpha[c] = 1.0f / (255.0f * std_[c]);
                beta[c] = -mean_[c] / std_[c];
            }
            Tensor t;
            if (!backend_->fused_preprocess(*image, &t, {dst_w, dst_h},
                                            0.0f, 0.0f, scale_x, scale_y,
                                            alpha, beta, false, 0.0f)) return false;
            tensors.emplace_back(std::move(t));
        }
        outputs->resize(1);
        (*outputs)[0] = Tensor::concat(tensors, 0);
        return true;
    }
} // namespace modeldeploy::vision::ocr
