//
// Created by aichao on 2025/3/26.
//
#pragma once

#include "csrc/base_model.h"
#include "csrc/core/md_decl.h"
#include "csrc/vision/face/face_det/scrfd.h"
#include "csrc/vision/face/face_as/face_as_first.h"
#include "csrc/vision/face/face_as/face_as_second.h"

namespace modeldeploy::vision::face {
    /*! @brief SCRFD model object used when to load a SCRFD model exported by SCRFD.
             */
    class MODELDEPLOY_CXX_EXPORT SeetaFaceAsPipeline : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
         *
         * \param[in] face_det_model_file Path of model file, e.g ./scrfd.onnx, loacal
         * \param[in] first_model_file Path of model file, e.g ./scrfd.onnx, loacal
         * \param[in] second_model_file Path of model file, e.g ./scrfd.onnx, global
         * \param[in] thread_num RuntimeOption for inference, the default will use cpu
         */
        explicit SeetaFaceAsPipeline(
            const std::string& face_det_model_file,
            const std::string& first_model_file,
            const std::string& second_model_file,
            int thread_num = 8);

        [[nodiscard]] std::string name() const override { return "face_pipeline"; }

        /** \brief Predict the face detection result for an input image
         *
         * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] results The output face FaceAntiSpoof result will be writen to this structure
         * \param fuse_threshold
         * \return true if the prediction successed, otherwise false
         */
        bool predict(cv::Mat& im, FaceAntiSpoofResult* results, float fuse_threshold = 0.8f, float clarity_threshold = 0.3);


        [[nodiscard]] bool is_initialized() const override;

    private:
        std::unique_ptr<SCRFD> face_det_ = nullptr;
        std::unique_ptr<SeetaFaceAntiSpoofFirst> face_as_first_ = nullptr;
        std::unique_ptr<SeetaFaceAntiSpoofSecond> face_as_second_ = nullptr;
    };
}
