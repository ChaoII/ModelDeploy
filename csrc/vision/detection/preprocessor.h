//
// Created by aichao on 2025/2/20.
//
#pragma once

#include <vector>
#include "core/tensor.h"
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision::detection {
    class MODELDEPLOY_CXX_EXPORT UltralyticsPreprocessor {
    public:
        UltralyticsPreprocessor();

        bool run(std::vector<ImageData>* images, std::vector<Tensor>* outputs,
                 std::vector<LetterBoxRecord>* letter_box_records) const;

        void set_size(const std::vector<int>& size) { size_ = size; }

        [[nodiscard]] std::vector<int> get_size() const { return size_; }

        void set_padding_value(const std::vector<float>& padding_value) {
            padding_value_ = padding_value;
        }

        void use_cuda_preproc() { use_cuda_preproc_ = true; };

        [[nodiscard]] std::vector<float> get_padding_value() const { return padding_value_; }

    protected:
        bool preprocess(ImageData* image, Tensor* output, LetterBoxRecord* letter_box_record) const;

        bool use_cuda_preproc_ = false;
        std::vector<int> size_;
        std::vector<float> padding_value_;
    };
} // namespace detection
