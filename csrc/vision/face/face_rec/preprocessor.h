//
// Created by aichao on 2025/3/24.
//

#pragma once
#include "core/md_decl.h"
#include "vision/common/result.h"
#include "core/tensor.h"
#include "vision/common/image_data.h"
#include "vision/processors/processor_factory.h"
#include "vision/processors/cpu/cpu_processor_backend.h"


namespace modeldeploy::vision::face {
    /*! @brief Preprocessor object for AdaFace serials model.
     */
    class MODELDEPLOY_CXX_EXPORT SeetaFaceIDPreprocessor {
    public:
        /** \brief Create a preprocessor instance for AdaFace serials model
         */
        SeetaFaceIDPreprocessor() = default;

        /** \brief Process the input image and prepare input tensors for runtime
         *
         * \param[in] images The input image data list, all the elements are returned by cv::imread()
         * \param[in] outputs The output tensors which will feed in runtime
         * \return true if the preprocess successful, otherwise false
         */
        bool run(std::vector<ImageData>* images, std::vector<Tensor>* outputs) const;

        /// Get Size
        std::vector<int> get_size() { return size_; }

        /// Set size.
        void set_size(const std::vector<int>& size) { size_ = size; }

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
        bool preprocess(ImageData* image, Tensor* output) const;
        // Argument for image preprocessing step, tuple of (width, height),
        // decide the target size after resize, default (248, 248)
        std::vector<int> size_{248, 248};
        std::shared_ptr<VisionProcessorBackend> backend_ =
            std::make_shared<CpuProcessorBackend>();
    };
} // namespace modeldeploy::vision::faceid
