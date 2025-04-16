//
// Created by aichao on 2025/3/26.
//
#pragma once

#include "csrc/base_model.h"
#include "csrc/core/md_decl.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision::face {
    /*! @brief SCRFD model object used when to load a SCRFD model exported by SCRFD.
             */
    class MODELDEPLOY_CXX_EXPORT SeetaFaceAsFirst : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
         *
         * \param[in] model_file Path of model file, e.g ./scrfd.onnx
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu
         */
        explicit SeetaFaceAsFirst(const std::string &model_file,
                                         const RuntimeOption &custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "face_as_first"; }

        /** \brief Predict the face detection result for an input image
         *
         * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output face FaceAntiSpoof result will be writen to this structure
         * \return true if the prediction successed, otherwise false
         */
        bool predict(cv::Mat& im, float *result);

        /// Argument for image preprocessing step, tuple of (width, height), decide the target size after resize, default (640, 640)
        std::vector<int> size_{224, 224};


    private:
        bool Initialize();

        bool preprocess(cv::Mat *mat, Tensor *output);

        static bool postprocess(const std::vector<Tensor> &infer_result, float *result);
    };
}
