//
// Created by aichao on 2025/3/21.
//

#pragma once
#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/ocr/utils/ocr_postprocess_op.h"
#include "csrc/vision/ocr/structurev2_layout_preprocessor.h"
#include "csrc/vision/ocr/structurev2_layout_postprocessor.h"


namespace modeldeploy::vision::ocr {
    /*! @brief StructureV2Layout object is used to load the PP-StructureV2-Layout detection model.
     */
    class MODELDEPLOY_CXX_EXPORT StructureV2Layout : public BaseModel {
    public:
        StructureV2Layout();
        /** \brief Set path of model file, and the configuration of runtime
         *
         * \param[in] model_file Path of model file, e.g ./picodet_lcnet_x1_0_fgd_layout_cdla_infer/model.pdmodel.
         * \param[in] params_file Path of parameter file, e.g ./picodet_lcnet_x1_0_fgd_layout_cdla_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
         * \param[in] model_format Model format of the loaded model, default is Paddle format.
         */
        StructureV2Layout(const std::string& model_file,
                          const RuntimeOption& custom_option = RuntimeOption());



        /// Get model's name
        std::string ModelName() const { return "pp-structurev2-layout"; }

        /** \brief DEPRECATED Predict the detection result for an input image
         *
         * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output detection result
         * \return true if the prediction successed, otherwise false
         */
        virtual bool Predict(cv::Mat* im, DetectionResult* result);

        /** \brief Predict the detection result for an input image
         * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output detection result
         * \return true if the prediction successed, otherwise false
         */
        virtual bool Predict(const cv::Mat& im, DetectionResult* result);

        /** \brief Predict the detection result for an input image list
         * \param[in] im The input image list, all the elements come from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] results The output detection result list
         * \return true if the prediction successed, otherwise false
         */
        virtual bool BatchPredict(const std::vector<cv::Mat>& imgs,
                                  std::vector<DetectionResult>* results);

        /// Get preprocessor reference ofStructureV2LayoutPreprocessor
        virtual StructureV2LayoutPreprocessor& GetPreprocessor() {
            return preprocessor_;
        }

        /// Get postprocessor reference ofStructureV2LayoutPostprocessor
        virtual StructureV2LayoutPostprocessor& GetPostprocessor() {
            return postprocessor_;
        }

    private:
        bool Initialize();
        StructureV2LayoutPreprocessor preprocessor_;
        StructureV2LayoutPostprocessor postprocessor_;
    };
} // namespace modeldeploy::vision::ocr
