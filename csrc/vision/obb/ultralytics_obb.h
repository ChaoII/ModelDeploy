//
// Created by aichao on 2025/5/30.
//

#pragma once

#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/obb/preprocessor.h"
#include "vision/obb/postprocessor.h"

namespace modeldeploy::vision::detection {
    /*! @brief YOLOv5Seg model object used when to load a YOLOv5Seg model exported by YOLOv5.
    */
    class MODELDEPLOY_CXX_EXPORT UltralyticsObb : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
        *
        * \param[in] model_file Path of model file, e.g ./yolov5seg.onnx
        * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
        */
        explicit UltralyticsObb(const std::string& model_file,
                                const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "UltralyticsObb"; }

        /** \brief Predict the detection result for an input image
        *
        * \param[in] image The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
        * \param[in] result The output detection result will be writen to this structure
        * \param timers
        * \return true if the prediction successed, otherwise false
        */
        virtual bool predict(const ImageData& image, std::vector<ObbResult>* result, TimerArray* timers = nullptr);

        /** \brief Predict the detection results for a batch of input images
        *
        * \param[in] images The input image list, each element comes from cv::imread()
        * \param[in] results The output detection result list
        * \param timers
        * \return true if the prediction successed, otherwise false
        */
        virtual bool batch_predict(const std::vector<ImageData>& images,
                                   std::vector<std::vector<ObbResult>>* results, TimerArray* timers = nullptr);

        /// Get preprocessor reference of YOLOv5Seg
        virtual UltralyticsObbPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        /// Get postprocessor reference of YOLOv5Seg
        virtual UltralyticsObbPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        UltralyticsObbPreprocessor preprocessor_;
        UltralyticsObbPostprocessor postprocessor_;
    };
}
