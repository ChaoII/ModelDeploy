//
// Created by aichao on 2025/2/20.
//
#pragma once
#include "../utils.h"
#include <map>

namespace modeldeploy::vision::detection {
    class YOLOv8Preprocessor {
    public:
        YOLOv8Preprocessor();

        bool run(std::vector<cv::Mat>* images, std::vector<MDTensor>* outputs,
                 std::vector<std::map<std::string, std::array<float, 2>>>* ims_info);

        void set_size(const std::vector<int>& size) { size_ = size; }

        std::vector<int> get_size() const { return size_; }

        void set_padding_value(const std::vector<float>& padding_value) {
            padding_value_ = padding_value;
        }

        std::vector<float> get_padding_value() const { return padding_value_; }

        void set_scale_up(bool is_scale_up) {
            is_scale_up_ = is_scale_up;
        }

        bool get_scale_up() const { return is_scale_up_; }

        void set_mini_pad(bool is_mini_pad) {
            is_mini_pad_ = is_mini_pad;
        }

        bool get_mini_pad() const { return is_mini_pad_; }

        void set_stride(int stride) {
            stride_ = stride;
        }

        bool get_stride() const { return stride_; }

    protected:
        bool preprocess(cv::Mat* mat, MDTensor* output,
                        std::map<std::string, std::array<float, 2>>* im_info);

        void letter_box(cv::Mat* mat);

        std::vector<int> size_;
        std::vector<float> padding_value_;
        bool is_mini_pad_;
        // while is_mini_pad = false and is_no_pad = true,
        // will resize the image to the set size
        bool is_no_pad_;

        // if is_scale_up is false, the input image only can be zoom out,
        // the maximum resize scale cannot exceed 1.0
        bool is_scale_up_;

        // padding stride, for is_mini_pad
        int stride_;

        // for offseting the boxes by classes when using NMS
        float max_wh_;
    };
} // namespace detection
