//
// Created by aichao on 2025/2/24.
//

#pragma once


#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/classification/preprocessor.h"
#include "vision/classification/postprocessor.h"

namespace modeldeploy::vision::classification {
    /*! @brief YOLOv5Cls model object used when to load a YOLOv5Cls model exported by YOLOv5Cls.
     */
    class MODELDEPLOY_CXX_EXPORT Classification : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
         *
         * \param[in] model_file Path of model file, e.g ./yolov5cls.onnx
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
         */
        explicit Classification(const std::string& model_file, const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "Classification"; }

        /** \brief Predict the classification result for an input image
         *
         * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output classification result will be writen to this structure
         * \return true if the prediction successed, otherwise false
         */
        virtual bool predict(const ImageData& img, ClassifyResult* result);

        /** \brief Predict the classification results for a batch of input images
         *
         * \param[in] images input image list, each element comes from cv::imread()
         * \param[in] results The output classification result list
         * \return true if the prediction successed, otherwise false
         */
        virtual bool batch_predict(const std::vector<ImageData>& images,
                                   std::vector<ClassifyResult>* results);

        /// Get preprocessor reference of YOLOv5Cls
        virtual ClassificationPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        /// Get postprocessor reference of YOLOv5Cls
        virtual ClassificationPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        ClassificationPreprocessor preprocessor_;
        ClassificationPostprocessor postprocessor_;
    };
} // namespace classification
