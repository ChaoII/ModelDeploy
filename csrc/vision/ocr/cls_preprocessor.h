//
// Created by aichao on 2025/2/21.
//

#pragma once



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
        bool run(const std::vector<ImageData>* images, std::vector<Tensor>* outputs,
                 size_t start_index, size_t end_index);

        /** \brief Implement the virtual function of ProcessorManager, Apply() is the
         *  body of Run(). Apply() contains the main logic of preprocessing, Run() is
         *  called by users to execute preprocessing
         *
         * \param[in] image_batch The input image batch
         * \param[in] outputs The output tensors which will feed in runtime
         * \return true if the preprocess successed, otherwise false
         */
        virtual bool apply(std::vector<ImageData>* image_batch, std::vector<Tensor>* outputs);

        /// Set preprocess normalize parameters, please call this API to customize
        /// the normalize parameters, otherwise it will use the default normalize
        /// parameters.
        void set_normalize(const std::vector<float>& mean,
                           const std::vector<float>& std,
                           bool is_scale) {
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

    private:

        std::vector<int> cls_image_shape_ = {3, 48, 192};
        std::vector<float> mean_{0.5f, 0.5f, 0.5f};
        std::vector<float> std_{0.5f, 0.5f, 0.5f};
        bool is_scale_ = true;
    };
} // namespace ocr
