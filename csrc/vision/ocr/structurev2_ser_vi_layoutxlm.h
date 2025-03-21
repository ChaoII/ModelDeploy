//
// Created by aichao on 2025/3/21.
//

#pragma once
#include "csrc/base_model.h"


/** \brief All classification model APIs are defined inside this namespace
 *
 */
namespace modeldeploy::vision::ocr {
    /*! @brief StructureV2SERViLayoutXLM model object used when to load a StructureV2SERViLayoutXLM model exported by StructureV2SERViLayoutXLMModel repository
     */
    class MODELDEPLOY_CXX_EXPORT StructureV2SERViLayoutXLMModel : public BaseModel {
    public:
        /** \brief Set path of model file and configuration file, and the configuration of runtime
         *
         * \param[in] model_file Path of model file, e.g ser_vi_layoutxlm/model.pdmodel
         * \param[in] params_file Path of parameter file, e.g ser_vi_layoutxlm/model.pdiparams, if the model format is ONNX, this parameter will be ignored
         * \param[in] config_file Path of configuration file for deployment, e.g ser_vi_layoutxlm/infer_cfg.yml
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
         * \param[in] model_format Model format of the loaded model, default is Paddle format
         */
        StructureV2SERViLayoutXLMModel(const std::string& model_file,
                                       const RuntimeOption& custom_option = RuntimeOption());


        /// Get model's name
        virtual std::string ModelName() const {
            return "StructureV2SERViLayoutXLMModel";
        }

    protected:
        bool Initialize();
    };
} // namespace modeldeploy::vision::ocr
