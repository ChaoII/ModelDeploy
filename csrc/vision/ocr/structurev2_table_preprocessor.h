//
// Created by aichao on 2025/3/21.
//

#pragma once
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/pad.h"
#include "csrc/vision/common/processors/normalize.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/result.h"
#include "opencv2/opencv.hpp"


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
         * \return true if the preprocess successed, otherwise false
         */
        bool Run(std::vector<cv::Mat>* images, std::vector<MDTensor>* outputs,
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
        virtual bool Apply(std::vector<cv::Mat>* image_batch, std::vector<MDTensor>* outputs);

        /// Get the image info of the last batch, return a list of array
        /// {image width, image height, resize width, resize height}
        const std::vector<std::array<int, 4>>* GetBatchImgInfo() {
            return &batch_det_img_info_;
        }

    private:
        void StructureV2TableResizeImage(cv::Mat* mat, int batch_idx);
        // for recording the switch of hwc2chw
        bool disable_permute_ = false;
        // for recording the switch of normalize
        bool disable_normalize_ = false;
        // for SLANet or SLANet_Plus max_len = 484,for SLANeXt_wired max_len=512
        int max_len = 484;
        std::vector<int> rec_image_shape_ = {3, max_len, max_len};
        bool static_shape_infer_ = false;
        std::shared_ptr<Resize> resize_op_;
        std::shared_ptr<Pad> pad_op_;
        std::shared_ptr<Normalize> normalize_op_;
        std::shared_ptr<HWC2CHW> hwc2chw_op_;
        std::vector<std::array<int, 4>> batch_det_img_info_;
    };
} // namespace modeldeploy::vision::ocr
