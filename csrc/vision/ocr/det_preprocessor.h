//
// Created by aichao on 2025/2/21.
//
#pragma once


namespace modeldeploy::vision::ocr {
    class MODELDEPLOY_CXX_EXPORT DBDetectorPreprocessor {
    public:
        virtual ~DBDetectorPreprocessor() = default;
        DBDetectorPreprocessor();

        virtual bool apply(const std::vector<ImageData>& image_batch,
                           std::vector<Tensor>* outputs);


        void use_cuda_preproc() { use_cuda_preproc_ = true; }

        void set_max_side_len(const int max_side_len) { max_side_len_ = max_side_len; }

        [[nodiscard]] int get_max_side_len() const { return max_side_len_; }

        void set_normalize(const std::vector<float>& mean,
                           const std::vector<float>& std) {
            mean_ = mean;
            std_ = std;
        }

        [[nodiscard]] std::vector<int> get_static_img_size() const { return static_img_size_; }

        void set_static_img_size(const std::vector<int>& static_img_size) {
            static_img_size_ = static_img_size;
        }

        const std::vector<std::array<int, 4>>* get_batch_img_info() const {
            return &batch_info_;
        }

    private:
        bool preprocess(const ImageData& image, Tensor* output, const std::vector<int>& resize_size,
                        const std::vector<int>& dst_size) const;


        std::array<int, 4> ocr_detector_get_info(const ImageData* image, int max_size_len) const;
        bool use_cuda_preproc_ = false;
        // for recording the switch of hwc2chw
        int max_side_len_ = 960;
        std::vector<int> static_img_size_ = {};
        std::vector<std::array<int, 4>> batch_info_;
        std::vector<float> mean_{0.485f, 0.456f, 0.406f};
        std::vector<float> std_{0.229f, 0.224f, 0.225f};
        float pad_value_ = 0.0f;
    };
}
