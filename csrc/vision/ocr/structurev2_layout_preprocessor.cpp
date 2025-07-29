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
        image->to_mat(&img);
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
        std::vector<cv::Mat> _images;
        for (size_t i = 0; i < image_batch->size(); ++i) {
            ImageData* image = &image_batch->at(i);
            cv::Mat mat;
            image->to_mat(&mat);
            batch_layout_img_info_[i] = get_layout_image_info(image);
            Resize::apply(&mat, batch_layout_img_info_[i][2], batch_layout_img_info_[i][3]);
            NormalizeAndPermute::apply(&mat, mean_, std_, is_scale_);
            _images.push_back(mat);
        }
        outputs->resize(1);
        utils::mats_to_tensor(_images, &(*outputs)[0]);
        return true;
    }
} // namespace modeldeploy::vision::ocr
