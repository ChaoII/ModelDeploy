//
// Created by aichao on 2025/2/21.
//


#pragma once
#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/ocr/utils/ocr_postprocess_op.h"
#include "csrc/vision/ocr/rec_preprocessor.h"
#include "csrc/vision/ocr/rec_postprocessor.h"

namespace modeldeploy::vision::ocr {
    /*! @brief Recognizer object is used to load the recognition model provided by PaddleOCR.
     */
    class MODELDEPLOY_CXX_EXPORT Recognizer : public BaseModel {
    public:
        Recognizer() = default;
        /** \brief Set path of model file, and the configuration of runtime
         *
         * \param[in] model_file Path of model file, e.g ./ch_PP-OCRv3_rec_infer/model.pdmodel.
         * \param[in] label_path Path of label file used by OCR recognition model. e.g ./ppocr_keys_v1.txt
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
         */
        explicit Recognizer(const std::string& model_file, const std::string& label_path = "",
                            const RuntimeOption& custom_option = RuntimeOption());

        /// Get model's name
        std::string name() const override { return "ppocr/ocr_rec"; }


        /** \brief Predict the input image and get OCR recognition model result.
         *
         * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] text The text result of rec model will be written into this parameter.
         * \param[in] rec_score The sccore result of rec model will be written into this parameter.
         * \return true if the prediction is successed, otherwise false.
         */
        virtual bool predict(const cv::Mat& img, std::string* text, float* rec_score);

        /** \brief Predict the input image and get OCR recognition model result.
         *
         * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] ocr_result The output of OCR recognition model result will be writen to this structure.
         * \return true if the prediction is successes, otherwise false.
         */
        virtual bool predict(const cv::Mat& img, OCRResult* ocr_result);

        /** \brief BatchPredict the input image and get OCR recognition model result.
         *
         * \param[in] images The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] ocr_result The output of OCR recognition model result will be writen to this structure.
         * \return true if the prediction is successes, otherwise false.
         */
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   OCRResult* ocr_result);

        /** \brief BatchPredict the input image and get OCR recognition model result.
         *
         * \param[in] images The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] texts The list of text results of rec model will be written into this vector.
         * \param[in] rec_scores The list of sccore result of rec model will be written into this vector.
         * \return true if the prediction is successed, otherwise false.
         */
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<std::string>* texts, std::vector<float>* rec_scores);

        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<std::string>* texts, std::vector<float>* rec_scores,
                                   size_t start_index, size_t end_index,
                                   const std::vector<int>& indices);

        /// Get preprocessor reference of DBDetectorPreprocessor
        virtual RecognizerPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        /// Get postprocessor reference of DBDetectorPostprocessor
        virtual RecognizerPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    private:
        bool initialize();
        RecognizerPreprocessor preprocessor_;
        RecognizerPostprocessor postprocessor_;
    };
}
