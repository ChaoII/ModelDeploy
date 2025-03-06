//
// Created by aichao on 2025/2/21.
//
#pragma once

#include "../common/processors/resize.h"
#include "../common/processors/pad.h"
#include "../common/processors/normalize_and_permute.h"


namespace modeldeploy::vision::ocr {
    class DBDetectorPreprocessor {
    public:
        virtual ~DBDetectorPreprocessor() = default;
        DBDetectorPreprocessor();

        virtual bool apply(std::vector<cv::Mat>* image_batch,
                           std::vector<MDTensor>* outputs);

        void set_max_side_len(int max_side_len) { max_side_len_ = max_side_len; }

        int get_max_side_len() const { return max_side_len_; }

        void set_normalize(const std::vector<float>& mean,
                           const std::vector<float>& std,
                           bool is_scale) {
            normalize_permute_op_ = std::make_shared<NormalizeAndPermute>(mean, std, is_scale);
        }

        const std::vector<std::array<int, 4>>* get_batch_img_info() {
            return &batch_det_img_info_;
        }


        void disable_normalize() { disable_permute_ = true; }

        void disable_permute() { disable_normalize_ = true; }

        void set_det_image_shape(const std::vector<int>& det_image_shape) {
            det_image_shape_ = det_image_shape;
        }

        std::vector<int> get_det_image_shape() const { return det_image_shape_; }

        void set_static_shape_infer(bool static_shape_infer) {
            static_shape_infer_ = static_shape_infer;
        }

        bool get_static_shape_infer() const { return static_shape_infer_; }

    private:
        bool resize_image(cv::Mat* img, int resize_w, int resize_h,
                          int max_resize_w, int max_resize_h);
        std::array<int, 4> ocr_detector_get_info(cv::Mat* img, int max_size_len);
        // for recording the switch of hwc2chw
        bool disable_permute_ = false;
        // for recording the switch of normalize
        bool disable_normalize_ = false;
        int max_side_len_ = 960;
        std::vector<std::array<int, 4>> batch_det_img_info_;
        std::shared_ptr<Resize> resize_op_;
        std::shared_ptr<Pad> pad_op_;
        std::shared_ptr<NormalizeAndPermute> normalize_permute_op_;
        std::vector<int> det_image_shape_ = {3, 960, 960};
        bool static_shape_infer_ = false;
    };
}
