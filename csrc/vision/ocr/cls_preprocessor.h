//
// Created by aichao on 2025/2/21.
//

#pragma once

#include "core/tensor.h"
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/processors/processor_factory.h"
#include "vision/processors/cpu/cpu_processor_backend.h"

namespace modeldeploy::vision::ocr {
    class MODELDEPLOY_CXX_EXPORT ClassifierPreprocessor {
    public:
        virtual ~ClassifierPreprocessor() = default;
        ClassifierPreprocessor();

        /** \brief Process the input image and prepare input tensors for runtime
         *
         * \param[in] images The input data list, all the elements are FDMat
         * \param[in] outputs The output tensors which will be fed into runtime
         * \param start_index
         * \param end_index
         * \return true if the preprocess successed, otherwise false
         */
        bool run(const std::vector<ImageData>& images, std::vector<Tensor>* outputs,
                 size_t start_index, size_t end_index);

        /** \brief Implement the virtual function of ProcessorManager, Apply() is the
         *  body of Run(). Apply() contains the main logic of preprocessing, Run() is
         *  called by users to execute preprocessing
         *
         * \param[in] image_batch The input image batch
         * \param[in] outputs The output tensors which will feed in runtime
         * \return true if the preprocess successed, otherwise false
         */
        virtual bool apply(const std::vector<ImageData>& image_batch, std::vector<Tensor>* outputs);

        /// Set preprocess normalize parameters, please call this API to customize
        /// the normalize parameters, otherwise it will use the default normalize
        /// parameters.
        void set_normalize(const std::vector<float>& mean,
                           const std::vector<float>& std,
                           const bool is_scale) {
            mean_ = mean;
            std_ = std;
            is_scale_ = is_scale;
        }

        /// Set cls_image_shape for the classification preprocess
        void set_cls_image_shape(const std::vector<int>& cls_image_shape) {
            cls_image_shape_ = cls_image_shape;
        }

        /// Get cls_image_shape for the classification preprocess
        [[nodiscard]] std::vector<int> get_cls_image_shape() const { return cls_image_shape_; }

        void use_cuda_preproc() {
            backend_ = create_processor_backend(Device::GPU, Backend::ORT, 0);
        }

        void set_processor_backend(std::shared_ptr<VisionProcessorBackend> backend) {
            backend_ = std::move(backend);
        }

        [[nodiscard]] std::shared_ptr<VisionProcessorBackend> get_processor_backend() const {
            return backend_;
        }

    private:
        // alpha/beta 由 mean/std/is_scale 决定（模型配置，batch 内共享）
        [[nodiscard]] std::vector<float> alpha() const {
            const float s = is_scale_ ? 255.0f : 1.0f;
            return {1.0f / (s * std_[0]), 1.0f / (s * std_[1]), 1.0f / (s * std_[2])};
        }
        [[nodiscard]] std::vector<float> beta() const {
            return {-mean_[0] / std_[0], -mean_[1] / std_[1], -mean_[2] / std_[2]};
        }

        std::vector<int> cls_image_shape_ = {3, 48, 192};
        std::vector<float> mean_{0.5f, 0.5f, 0.5f};
        std::vector<float> std_{0.5f, 0.5f, 0.5f};
        bool is_scale_ = true;
        std::shared_ptr<VisionProcessorBackend> backend_ =
            std::make_shared<CpuProcessorBackend>();
    };
} // namespace ocr
