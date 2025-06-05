//
// Created by aichao on 2025/3/24.
//

#pragma once

#include "csrc/base_model.h"
#include "csrc/vision/face/face_rec/postprocessor.h"
#include "csrc/vision/face/face_rec/preprocessor.h"


namespace modeldeploy::vision::face
{
    /*! @brief AdaFace model object used when to load a AdaFace model exported by AdaFace.
     */
    class MODELDEPLOY_CXX_EXPORT SeetaFaceID : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
         *
         * \param[in] model_file Path of model file, e.g ./adaface.onnx
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
         */
        explicit SeetaFaceID(const std::string& model_file, const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "SeetaFaceID"; }

        /** \brief Predict the detection result for an input image
         *
         * \param[in] image The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output FaceRecognitionResult will be writen to this structure
         * \param timers
         * \return true if the prediction successed, otherwise false
         */
        virtual bool predict(const cv::Mat& image, FaceRecognitionResult* result, TimerArray* timers = nullptr);

        /** \brief Predict the detection results for a batch of input images
         *
         * \param[in] images The input image list, each element comes from cv::imread()
         * \param[in] results The output FaceRecognitionResult list
         * \param timers
         * \return true if the prediction successed, otherwise false
         */
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<FaceRecognitionResult>* results, TimerArray* timers = nullptr);

        /// Get preprocessor reference of AdaFace
        virtual SeetaFaceIDPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        /// Get postprocessor reference of AdaFace
        virtual SeetaFaceIDPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        SeetaFaceIDPreprocessor preprocessor_;
        SeetaFaceIDPostprocessor postprocessor_;
    };
} // namespace modeldeploy::vision::faceid
