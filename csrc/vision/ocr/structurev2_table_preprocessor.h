//
// Created by aichao on 2025/3/21.
//

#pragma once



namespace modeldeploy::vision::ocr {
    /*! @brief Preprocessor object for table model.
     */
    class MODELDEPLOY_CXX_EXPORT StructureV2TablePreprocessor {
    public:
        StructureV2TablePreprocessor();
        /** \brief Process the input image and prepare input tensors for runtime
         *
         * \param[in] images The input data list, all the elements are FDMat
         * \param[in] outputs The output tensors which will be fed into runtime
         * \param start_index
         * \param end_index
         * \param indices
         * \return true if the preprocess successed, otherwise false
         */
        bool run(std::vector<ImageData>* images, std::vector<Tensor>* outputs,
                 size_t start_index, size_t end_index,
                 const std::vector<int>& indices);

        /** \brief Implement the virtual function of ProcessorManager, Apply() is the
         *  body of Run(). Apply() contains the main logic of preprocessing, Run() is
         *  called by users to execute preprocessing
         *
         * \param[in] image_batch The input image batch
         * \param[in] outputs The output tensors which will feed in runtime
         * \return true if the preprocess successed, otherwise false
         */
        bool run(std::vector<ImageData>* image_batch, std::vector<Tensor>* outputs);

        /// Get the image info of the last batch, return a list of array
        /// {image width, image height, resize width, resize height}
        const std::vector<std::array<int, 4>>* GetBatchImgInfo() {
            return &batch_det_img_info_;
        }

    private:
        // for SLANet or SLANet_Plus max_len = 484, for SLANeXt_wired max_len = 512
        int max_len = 512;
        std::vector<int> rec_image_shape_ = {3, max_len, max_len};
        bool static_shape_infer_ = false;
        std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
        std::vector<float> std_ = {0.229f, 0.224f, 0.225f};
        std::vector<float> pad_value_ = {0.0f, 0.0f, 0.0f};
        bool is_scale_ = true;
        std::vector<std::array<int, 4>> batch_det_img_info_;
    };
} // namespace modeldeploy::vision::ocr
