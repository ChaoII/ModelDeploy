//
// Created by aichao on 2025/2/24.
//

#pragma once

#include "csrc/base_model.h"
#include "csrc/vision/classification/preprocessor.h"
#include "csrc/vision/classification/postprocessor.h"

namespace modeldeploy::vision::classification {
    /*! @brief YOLOv5Cls model object used when to load a YOLOv5Cls model exported by YOLOv5Cls.
     */
    class MODELDEPLOY_CXX_EXPORT UltralyticsCls : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
         *
         * \param[in] model_file Path of model file, e.g ./yolov5cls.onnx
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
         */
        explicit UltralyticsCls(const std::string& model_file, const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "UltralyticsCls"; }

        /** \brief Predict the classification result for an input image
         *
         * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output classification result will be writen to this structure
         * \return true if the prediction successed, otherwise false
         */
        virtual bool predict(const cv::Mat& img, ClassifyResult* result);

        /** \brief Predict the classification results for a batch of input images
         *
         * \param[in] images, The input image list, each element comes from cv::imread()
         * \param[in] results The output classification result list
         * \return true if the prediction successed, otherwise false
         */
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<ClassifyResult>* results);

        /// Get preprocessor reference of YOLOv5Cls
        virtual UltralyticsClsPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        /// Get postprocessor reference of YOLOv5Cls
        virtual UltralyticsClsPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        UltralyticsClsPreprocessor preprocessor_;
        UltralyticsClsPostprocessor postprocessor_;
    };
} // namespace classification
