//
// Created by aichao on 2025/2/20.
//
#pragma once

#include "core/md_decl.h"
#include "core/tensor.h"
#include "vision/common/struct.h"
#include "vision/common/image_data.h"
#include "vision/processors/processor_factory.h"
#include "vision/processors/cpu/cpu_processor_backend.h"

namespace modeldeploy::vision::lpr {
    class MODELDEPLOY_CXX_EXPORT LprDetPreprocessor {
    public:
        LprDetPreprocessor();

        bool run(std::vector<ImageData>* images, std::vector<Tensor>* outputs,
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

        void use_cuda_preproc() {
            backend_ = create_processor_backend(Device::GPU, Backend::ORT, 0);
        }

        void set_processor_backend(std::shared_ptr<VisionProcessorBackend> backend) {
            backend_ = std::move(backend);
        }

        [[nodiscard]] std::shared_ptr<VisionProcessorBackend> get_processor_backend() const {
            return backend_;
        }

    protected:
        bool preprocess(const ImageData* image, Tensor* output, LetterBoxRecord* letter_box_record) const;

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
        std::shared_ptr<VisionProcessorBackend> backend_ =
            std::make_shared<CpuProcessorBackend>();
    };
} // namespace detection
