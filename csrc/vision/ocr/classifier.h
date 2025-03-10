//
// Created by aichao on 2025/2/21.
//

#pragma once
#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"
#include "utils/ocr_postprocess_op.h"
#include "cls_postprocessor.h"
#include "cls_preprocessor.h"


namespace modeldeploy::vision::ocr {
    class MODELDEPLOY_CXX_EXPORT Classifier : public BaseModel {
    public:
        Classifier();
        /** \brief Set path of model file, and the configuration of runtime
         *
         * \param[in] model_file Path of model file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdmodel.
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
         */
        Classifier(const std::string& model_file, const RuntimeOption& custom_option = RuntimeOption());


        /// Get model's name
        std::string name() const override { return "ppocr/ocr_cls"; }

        /** \brief Predict the input image and get OCR classification model cls_result.
         *
         * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] cls_label The label result of cls model will be written in to this param.
         * \param[in] cls_score The score result of cls model will be written in to this param.
         * \return true if the prediction is successed, otherwise false.
         */
        virtual bool predict(const cv::Mat& img, int32_t* cls_label, float* cls_score);

        /** \brief Predict the input image and get OCR recognition model result.
         *
         * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] ocr_result The output of OCR recognition model result will be writen to this structure.
         * \return true if the prediction is successes, otherwise false.
         */
        virtual bool predict(const cv::Mat& img, OCRResult* ocr_result);

        /** \brief BatchPredict the input image and get OCR classification model result.
         *
         * \param[in] images The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] ocr_result The output of OCR classification model result will be writen to this structure.
         * \return true if the prediction is successes, otherwise false.
         */
        virtual bool batch_predict(const std::vector<cv::Mat>& images, OCRResult* ocr_result);

        /** \brief BatchPredict the input image and get OCR classification model cls_result.
         *
         * \param[in] images The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] cls_labels The label results of cls model will be written in to this vector.
         * \param[in] cls_scores The score results of cls model will be written in to this vector.
         * \return true if the prediction is successes, otherwise false.
         */
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<int32_t>* cls_labels,
                                   std::vector<float>* cls_scores);
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<int32_t>* cls_labels,
                                   std::vector<float>* cls_scores,
                                   size_t start_index, size_t end_index);

        /// Get preprocessor reference of ClassifierPreprocessor
        virtual ClassifierPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        /// Get postprocessor reference of ClassifierPostprocessor
        virtual ClassifierPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    private:
        bool initialize();
        ClassifierPreprocessor preprocessor_;
        ClassifierPostprocessor postprocessor_;
    };
} // namespace ocr
