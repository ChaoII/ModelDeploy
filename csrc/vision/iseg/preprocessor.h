//
// Created by aichao on 2025/4/14.
//

#pragma once

#include "core/tensor.h"
#include "vision/common/struct.h"
#include "vision/processors/processor_factory.h"
#include "vision/processors/cpu/cpu_processor_backend.h"

namespace modeldeploy::vision::detection {
    /*! @brief Preprocessor object for YOLOv5Seg serials model.
    */
    class MODELDEPLOY_CXX_EXPORT UltralyticsSegPreprocessor {
    public:
        /// Create a preprocessor instance for YOLOv5Seg serials model
        UltralyticsSegPreprocessor();
        /** \brief Process the input image and prepare input tensors for runtime
        *
        * \param[in] images The input image data list, all the elements are returned by cv::imread()
        * \param[in] outputs The output tensors which will feed in runtime
        * \param[in]  letter_box_records The shape info list, record input_shape and output_shape
        * \return true if the preprocess successed, otherwise false
        */
        bool run(const std::vector<ImageData>& images, std::vector<Tensor>* outputs,
                 std::vector<LetterBoxRecord>* letter_box_records) const;

        /// Set target size, tuple of (width, height), default size = {640, 640}
        void set_size(const std::vector<int>& size) { size_ = size; }

        /// Get target size, tuple of (width, height), default size = {640, 640}
        [[nodiscard]] std::vector<int> get_size() const { return size_; }

        /// Set padding value, size should be the same as channels
        void set_padding_value(const std::vector<float>& padding_value) {
            padding_value_ = padding_value;
        }

        /// Get padding value, size should be the same as channels
        [[nodiscard]] std::vector<float> get_padding_value() const { return padding_value_; }

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
        bool preprocess(const ImageData& image, Tensor* output,
                        LetterBoxRecord* letter_box_record) const;

        std::shared_ptr<VisionProcessorBackend> backend_ =
            std::make_shared<CpuProcessorBackend>();
        std::vector<int> size_;
        std::vector<float> padding_value_;
    };
}
