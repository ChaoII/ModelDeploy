//
// Created by aichao on 2025/4/14.
//

#pragma once

#include "csrc/base_model.h"
#include "csrc/vision/iseg/preprocessor.h"
#include "csrc/vision/iseg/postprocessor.h"

namespace modeldeploy::vision::detection {
    /*! @brief YOLOv5Seg model object used when to load a YOLOv5Seg model exported by YOLOv5.
    */
    class MODELDEPLOY_CXX_EXPORT UltralyticsSeg : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
        *
        * \param[in] model_file Path of model file, e.g ./yolov5seg.onnx
        * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
        */
        explicit UltralyticsSeg(const std::string& model_file,
                                const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "UltralyticsSeg"; }

        /** \brief Predict the detection result for an input image
        *
        * \param[in] image The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
        * \param[in] result The output detection result will be writen to this structure
        * \return true if the prediction successed, otherwise false
        */
        virtual bool predict(const cv::Mat& image, std::vector<InstanceSegResult>* result);

        /** \brief Predict the detection results for a batch of input images
        *
        * \param[in] images The input image list, each element comes from cv::imread()
        * \param[in] results The output detection result list
        * \return true if the prediction successed, otherwise false
        */
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<std::vector<InstanceSegResult>>* results);

        /// Get preprocessor reference of YOLOv5Seg
        virtual UltralyticsSegPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        /// Get postprocessor reference of YOLOv5Seg
        virtual UltralyticsSegPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        UltralyticsSegPreprocessor preprocessor_;
        UltralyticsSegPostprocessor postprocessor_;
    };
}
