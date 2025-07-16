//
// Created by aichao on 2025/3/21.
//

#pragma once

#include "base_model.h"
#include "vision/common/result.h"
#include "vision/ocr/utils/ocr_postprocess_op.h"
#include "vision/ocr/structurev2_layout_preprocessor.h"
#include "vision/ocr/structurev2_layout_postprocessor.h"


namespace modeldeploy::vision::ocr {
    /*! @brief StructureV2Layout object is used to load the PP-StructureV2-Layout detection model.
     */
    class MODELDEPLOY_CXX_EXPORT StructureV2Layout : public BaseModel {
    public:
        StructureV2Layout() = default;

        /** \brief Set path of model file, and the configuration of runtime
         *
         * \param[in] model_file Path of model file, e.g ./picodet_lcnet_x1_0_fgd_layout_cdla_infer/model.pdmodel.
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
         */
        explicit StructureV2Layout(const std::string& model_file,
                                   const RuntimeOption& custom_option = RuntimeOption());


        /// Get model's name
        [[nodiscard]] std::string name() const override { return "pp-structurev2-layout"; }

        /** \brief DEPRECATED Predict the detection result for an input image
         *
         * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output detection result
         * \return true if the prediction successed, otherwise false
         */
        virtual bool predict(cv::Mat* im, std::vector<DetectionResult>* result);

        /** \brief Predict the detection result for an input image
         * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output detection result
         * \return true if the prediction successed, otherwise false
         */
        virtual bool predict(const cv::Mat& im, std::vector<DetectionResult>* result);

        /** \brief Predict the detection result for an input image list
         * \param[in] images The input image list, all the elements come from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] results The output detection result list
         * \return true if the prediction successfully, otherwise false
         */
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<std::vector<DetectionResult>>* results);

        /// Get preprocessor reference ofStructureV2LayoutPreprocessor
        virtual StructureV2LayoutPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        /// Get postprocessor reference ofStructureV2LayoutPostprocessor
        virtual StructureV2LayoutPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    private:
        bool Initialize();

        StructureV2LayoutPreprocessor preprocessor_;
        StructureV2LayoutPostprocessor postprocessor_;
    };
} // namespace modeldeploy::vision::ocr
