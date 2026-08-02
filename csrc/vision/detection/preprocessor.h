//
// Created by aichao on 2025/2/20.
//
#pragma once

#include <vector>
#include "core/tensor.h"
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/common/image_data.h"
#include "vision/processors/processor_factory.h"
#include "vision/processors/cpu/cpu_processor_backend.h"

namespace modeldeploy::vision::detection {
    class MODELDEPLOY_CXX_EXPORT UltralyticsPreprocessor {
    public:
        UltralyticsPreprocessor();

        bool run(const std::vector<ImageData>& images, std::vector<Tensor>* outputs,
                 std::vector<LetterBoxRecord>* letter_box_records) const;


        bool run(const uint8_t* src_y,
                 const uint8_t* src_uv,
                 const std::vector<int>& src_size,
                 int step_y,
                 int step_uv,
                 Tensor* output,
                 LetterBoxRecord* letter_box_record) const;


        void set_size(const std::vector<int>& size) { size_ = size; }

        [[nodiscard]] std::vector<int> get_size() const { return size_; }

        void set_padding_value(const std::vector<float>& padding_value) {
            padding_value_ = padding_value;
        }

        void use_cuda_preproc() {
            backend_ = create_processor_backend(Device::GPU, Backend::ORT, 0);
        }

        void set_processor_backend(std::shared_ptr<VisionProcessorBackend> backend) {
            backend_ = std::move(backend);
        }

        [[nodiscard]] std::shared_ptr<VisionProcessorBackend> get_processor_backend() const {
            return backend_;
        }

        [[nodiscard]] std::vector<float> get_padding_value() const { return padding_value_; }

    protected:
        bool preprocess(const ImageData& image, Tensor* output, LetterBoxRecord* letter_box_record) const;

        std::shared_ptr<VisionProcessorBackend> backend_ =
            std::make_shared<CpuProcessorBackend>();
        std::vector<int> size_;
        std::vector<float> padding_value_;
    };
} // namespace detection
