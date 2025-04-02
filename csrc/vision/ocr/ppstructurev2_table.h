//
// Created by aichao on 2025/3/21.
//

#pragma once

#include <vector>
#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/ocr/structurev2_table.h"
#include "csrc/vision/ocr/dbdetector.h"
#include "csrc/vision/ocr/recognizer.h"
#include "csrc/vision/ocr/utils/ocr_postprocess_op.h"


/** \brief This pipeline can launch detection model, classification model and recognition model sequentially. All OCR pipeline APIs are defined inside this namespace.
 *
 */
namespace modeldeploy::pipeline {
    /*! @brief PPStructureV2Table is used to load PP-OCRv2 series models provided by PaddleOCR.
     */
    class MODELDEPLOY_CXX_EXPORT PPStructureV2Table : public BaseModel {
    public:
        /** \brief Set up the detection model path, recognition model path and table model path respectively.
         *
         * \param[in] det_model Path of detection model, e.g ./ch_PP-OCRv2_det_infer
         * \param[in] rec_model Path of recognition model, e.g ./ch_PP-OCRv2_rec_infer
         * \param[in] table_model Path of table recognition model, e.g ./en_ppstructure_mobile_v2.0_SLANet_infer
         */
        PPStructureV2Table(modeldeploy::vision::ocr::DBDetector *det_model,
                           modeldeploy::vision::ocr::Recognizer *rec_model,
                           modeldeploy::vision::ocr::StructureV2Table *table_model);


        /** \brief Predict the input image and get OCR result.
         *
         * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] result The output OCR result will be writen to this structure.
         * \return true if the prediction successed, otherwise false.
         */
        virtual bool predict(cv::Mat *img, modeldeploy::vision::OCRResult *result);

        virtual bool predict(const cv::Mat &img, modeldeploy::vision::OCRResult *result);

        /** \brief BatchPredict the input image and get OCR result.
         *
         * \param[in] images The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] batch_result The output list of OCR result will be writen to this structure.
         * \return true if the prediction successed, otherwise false.
         */
        virtual bool batch_predict(const std::vector<cv::Mat> &images,
                                   std::vector<modeldeploy::vision::OCRResult> *batch_result);

        bool set_rec_batch_size(int rec_batch_size);

        [[nodiscard]] int get_rec_batch_size() const;

    protected:
        modeldeploy::vision::ocr::DBDetector *detector_ = nullptr;
        modeldeploy::vision::ocr::Recognizer *recognizer_ = nullptr;
        modeldeploy::vision::ocr::StructureV2Table *table_ = nullptr;

    private:
        int rec_batch_size_ = 6;
    };

} // namespace modeldeploy::pipeline
