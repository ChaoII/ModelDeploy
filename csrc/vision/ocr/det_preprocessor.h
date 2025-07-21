//
// Created by aichao on 2025/2/21.
//
#pragma once




namespace modeldeploy::vision::ocr {
    class MODELDEPLOY_CXX_EXPORT DBDetectorPreprocessor {
    public:
        virtual ~DBDetectorPreprocessor() = default;
        DBDetectorPreprocessor();

        virtual bool apply(std::vector<ImageData>* image_batch,
                           std::vector<Tensor>* outputs);

        void set_max_side_len(int max_side_len) { max_side_len_ = max_side_len; }

        [[nodiscard]] int get_max_side_len() const { return max_side_len_; }

        void set_normalize(const std::vector<float>& mean,
                           const std::vector<float>& std,
                           bool is_scale) {
            mean_ = mean;
            std_ = std;
            is_scale_ = is_scale;
        }

        const std::vector<std::array<int, 4>>* get_batch_img_info() {
            return &batch_det_img_info_;
        }


        void set_det_image_shape(const std::vector<int>& det_image_shape) {
            det_image_shape_ = det_image_shape;
        }

        std::vector<int> get_det_image_shape() const { return det_image_shape_; }

        void set_static_shape_infer(bool static_shape_infer) {
            static_shape_infer_ = static_shape_infer;
        }

        bool get_static_shape_infer() const { return static_shape_infer_; }

    private:
        bool resize_image(ImageData* image, int resize_w, int resize_h,
                          int max_resize_w, int max_resize_h) const;
        std::array<int, 4> ocr_detector_get_info(const ImageData* image, int max_size_len) const;
        // for recording the switch of hwc2chw
        int max_side_len_ = 960;
        std::vector<std::array<int, 4>> batch_det_img_info_;
        std::vector<int> det_image_shape_ = {3, 960, 960};
        bool static_shape_infer_ = false;
        std::vector<float> mean_{0.5f, 0.5f, 0.5f};
        std::vector<float> std_{1.5f, 1.5f, 0.5f};
        std::vector<float> pad_value_ = {0.0f, 0.0f, 0.0f};
        bool is_scale_ = false;
    };
}
