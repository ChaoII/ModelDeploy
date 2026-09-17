//
// Created by aichao on 2025/3/24.
//

#pragma once

#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/face/face_age/postprocessor.h"
#include "vision/face/face_age/preprocessor.h"


namespace modeldeploy::vision::face {
    /*! @brief AdaFace model object used when to load a AdaFace model exported by AdaFace.
     */
    class MODELDEPLOY_CXX_EXPORT SeetaFaceAge : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
         *
         * \param[in] model_file Path of model file, e.g ./adaface.onnx
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
         */
        explicit SeetaFaceAge(const std::string& model_file, const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "seetaface age predict"; }

        [[nodiscard]] std::unique_ptr<SeetaFaceAge> clone() const;


        /** \brief Predict the detection result for an input image
         *
         * \param[in] image The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] age The output age will be writen to this structure
         * \return true if the prediction successed, otherwise false
         */
        virtual bool predict(const ImageData& image, int* age, TimerArray* timers = nullptr);

        /** \brief 预测年龄概率分布（新增，附属于既有接口；下标=年龄 0..N-1，和为 1）
         *
         * \param[in] image 输入人脸图（建议为对齐/裁剪后的人脸）
         * \param[out] dist 年龄分布
         * \return true 成功
         */
        virtual bool predict_dist(const ImageData& image, std::vector<float>* dist,
                                  TimerArray* timers = nullptr);

        /** \brief Predict the detection results for a batch of input images
         *
         * \param[in] images The input image list, each element comes from cv::imread()
         * \param[in] ages The output age list
         * \return true if the prediction successed, otherwise false
         */
        virtual bool batch_predict(const std::vector<ImageData>& images,
                                   std::vector<int>* ages, TimerArray* timers = nullptr);

        /// Get preprocessor reference of AdaFace
        virtual SeetaFaceAgePreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        /// Get postprocessor reference of AdaFace
        virtual SeetaFaceAgePostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        SeetaFaceAgePreprocessor preprocessor_;
        SeetaFaceAgePostprocessor postprocessor_;
    };
} // namespace modeldeploy::vision::faceid
