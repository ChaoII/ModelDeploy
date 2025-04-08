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
    class MODELDEPLOY_CXX_EXPORT SeetaFaceAsSecond : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
         *
         * \param[in] model_file Path of model file, e.g ./scrfd.onnx
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
         */
        explicit SeetaFaceAsSecond(const std::string &model_file,
                                          const RuntimeOption &custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "face_as_second"; }

        /** \brief Predict the face detection result for an input image
         *
         * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output face detection result will be writen to this structure
         * \return true if the prediction successed, otherwise false
         */
        bool predict(cv::Mat& im, std::vector<std::tuple<int, float>> *result);

        /// Argument for image preprocessing step, tuple of (width, height), decide the target size after resize, default (640, 640)
        std::vector<int> size_{300, 300};

    private:
        bool Initialize();

        bool preprocess(cv::Mat *mat, MDTensor *output);

        bool postprocess(const std::vector<MDTensor> &infer_result, std::vector<std::tuple<int, float>> *result);
    };
}
