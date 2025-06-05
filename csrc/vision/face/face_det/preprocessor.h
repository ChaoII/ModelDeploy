//
// Created by aichao on 2025/2/20.
//
#pragma once
#include "csrc/core/md_decl.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/common/struct.h"

namespace modeldeploy::vision::face {
    class MODELDEPLOY_CXX_EXPORT ScrfdPreprocessor {
    public:
        ScrfdPreprocessor();

        bool run(std::vector<cv::Mat>* images, std::vector<Tensor>* outputs,
                 std::vector<LetterBoxRecord>* letter_box_records) const;

        void set_size(const std::vector<int>& size) { size_ = size; }

        [[nodiscard]] std::vector<int> get_size() const { return size_; }

        void set_padding_value(const std::vector<float>& padding_value) {
            padding_value_ = padding_value;
        }

        [[nodiscard]] std::vector<float> get_padding_value() const { return padding_value_; }

        void set_scale_up(const bool is_scale_up) {
            is_scale_up_ = is_scale_up;
        }

        [[nodiscard]] bool get_scale_up() const { return is_scale_up_; }

        void set_mini_pad(const bool is_mini_pad) {
            is_mini_pad_ = is_mini_pad;
        }

        [[nodiscard]] bool get_mini_pad() const { return is_mini_pad_; }

        void set_stride(const int stride) {
            stride_ = stride;
        }

        [[nodiscard]] bool get_stride() const { return stride_; }

    protected:
        bool preprocess(cv::Mat* mat, Tensor* output, LetterBoxRecord* letter_box_record) const;

        std::vector<int> size_{640, 640};
        /// padding value, size should be the same as channels
        std::vector<float> padding_value_{0.0, 0.0, 0.0};
        /// only pad to the minimum rectangle which height and width is times of stride
        bool is_mini_pad_ = false;
        /// while is_mini_pad = false and is_no_pad = true,
        /// will resize the image to the set size
        bool is_no_pad_ = false;
        /// if is_scale_up is false, the input image only can be zoom out,
        /// the maximum resize scale cannot exceed 1.0
        bool is_scale_up_ = true;
        /// padding stride, for is_mini_pad
        int stride_ = 32;

    };
} // namespace detection
