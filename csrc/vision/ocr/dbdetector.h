//
// Created by aichao on 2025/2/21.
//

#pragma once
#include "base_model.h"
#include "vision/common/result.h"
#include "vision/ocr/det_postprocessor.h"
#include "vision/ocr/det_preprocessor.h"


namespace modeldeploy::vision::ocr {
    class MODELDEPLOY_CXX_EXPORT DBDetector : public BaseModel {
    public:
        DBDetector();
        /** \brief Set path of model file, and the configuration of runtime
         *
         * \param[in] model_file Path of model file, e.g ./ch_PP-OCRv3_det_infer/model.pdmodel.
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
         */
        explicit DBDetector(const std::string& model_file, const RuntimeOption& custom_option = RuntimeOption());


        /// Get model's name
        [[nodiscard]] std::string name() const override { return "ppocr/ocr_det"; }

        /** \brief Predict the input image and get OCR detection model result.
        *
        * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
        * \param[in] boxes_result The output of OCR detection model result will be writen to this structure.
        * \return true if the prediction is successed, otherwise false.
        */
        virtual bool predict(const cv::Mat& img,
                             std::vector<std::array<int, 8>>* boxes_result);

        /** \brief Predict the input image and get OCR detection model result.
         *
         * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] ocr_result The output of OCR detection model result will be writen to this structure.
         * \return true if the prediction is successes, otherwise false.
         */
        virtual bool predict(const cv::Mat& img, OCRResult* ocr_result);

        /** \brief BatchPredict the input image and get OCR detection model result.
        *
        * \param[in] images The list input of image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
        * \param[in] det_results The output of OCR detection model result will be writen to this structure.
        * \return true if the prediction is successed, otherwise false.
        */
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<std::vector<std::array<int, 8>>>* det_results);

        /** \brief BatchPredict the input image and get OCR detection model result.
         *
         * \param[in] images The list input of image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] ocr_results The output of OCR detection model result will be writen to this structure.
         * \return true if the prediction is successed, otherwise false.
         */
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<vision::OCRResult>* ocr_results);

        /// Get preprocessor reference of DBDetectorPreprocessor
        virtual DBDetectorPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        /// Get postprocessor reference of DBDetectorPostprocessor
        virtual DBDetectorPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    private:
        bool initialize();
        DBDetectorPreprocessor preprocessor_;
        DBDetectorPostprocessor postprocessor_;
    };
} // namespace modeldeploy
