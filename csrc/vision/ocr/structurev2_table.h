//
// Created by aichao on 2025/3/21.
//

#pragma once

#include "base_model.h"
#include "vision/common/result.h"
#include "vision/common/image_data.h"
#include "vision/ocr/structurev2_table_postprocessor.h"
#include "vision/ocr/structurev2_table_preprocessor.h"


/** \brief All OCR series model APIs are defined inside this namespace
 *
 */
namespace modeldeploy::vision::ocr {
    /*! @brief DBDetector object is used to load the detection model provided by PaddleOCR.
     */
    class MODELDEPLOY_CXX_EXPORT StructureV2Table : public BaseModel {
    public:
        StructureV2Table();

        /** \brief Set path of model file, and the configuration of runtime
         *
         * \param[in] model_file Path of model file, e.g ./en_ppstructure_mobile_v2.0_SLANet_infer/model.pdmodel.
         * \param table_char_dict_path
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
         */
        StructureV2Table(const std::string& model_file,
                         const std::string& table_char_dict_path = "",
                         const RuntimeOption& custom_option = RuntimeOption());


        /// Get model's name
        [[nodiscard]] std::string name() const override { return "ppocr/ocr_table"; }

        /** \brief Predict the input image and get OCR detection model result.
        *
        * \param[in] image The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
        * \param[in] boxes_result The output of OCR detection model result will be writen to this structure.
        * \param structure_result
        * \return true if the prediction is successed, otherwise false.
        */
        virtual bool predict(const ImageData& image,
                             std::vector<std::array<int, 8>>* boxes_result,
                             std::vector<std::string>* structure_result);

        /** \brief Predict the input image and get OCR detection model result.
         *
         * \param[in] image The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] ocr_result The output of OCR detection model result will be writen to this structure.
         * \return true if the prediction is successed, otherwise false.
         */
        virtual bool predict(const ImageData& image, vision::OCRResult* ocr_result);

        /** \brief BatchPredict the input image and get OCR detection model result.
        *
        * \param[in] images The list input of image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
        * \param[in] det_results The output of OCR detection model result will be writen to this structure.
        * \return true if the prediction is successed, otherwise false.
        */
        virtual bool batch_predict(const std::vector<ImageData>& images,
                                   std::vector<std::vector<std::array<int, 8>>>* det_results,
                                   std::vector<std::vector<std::string>>* structure_results);

        /** \brief BatchPredict the input image and get OCR detection model result.
         *
         * \param[in] images The list input of image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] ocr_results The output of OCR detection model result will be writen to this structure.
         * \return true if the prediction is successed, otherwise false.
         */
        virtual bool batch_predict(const std::vector<ImageData>& images,
                                   std::vector<vision::OCRResult>* ocr_results);

        /// Get preprocessor reference of StructureV2TablePreprocessor
        virtual StructureV2TablePreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        /// Get postprocessor reference of StructureV2TablePostprocessor
        virtual StructureV2TablePostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    private:
        bool initialize();

        StructureV2TablePreprocessor preprocessor_;
        StructureV2TablePostprocessor postprocessor_;
    };
} // namespace modeldeploy::vision::ocr
