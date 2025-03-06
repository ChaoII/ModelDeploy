//
// Created by aichao on 2025/2/21.
//

#pragma once

#include <vector>

#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"
#include "classifier.h"
#include "dbdetector.h"
#include "recognizer.h"
#include "./utils/ocr_postprocess_op.h"

namespace modeldeploy::vision::ocr {
    class PPOCRv4 : public BaseModel {
    public:
        PPOCRv4(const std::string& det_model_path,
                const std::string& cls_model_path,
                const std::string& rec_model_path,
                const std::string& dict_path,
                int thread_num = 8,
                int max_side_len = 960,
                double det_db_thresh = 0.3,
                double det_db_box_thresh = 0.6,
                double det_db_unclip_ratio = 1.5,
                const std::string& det_db_score_mode = "slow",
                bool use_dilation = false,
                int rec_batch_size = 8);

        ~PPOCRv4();

        /** \brief Predict the input image and get OCR result.
         *
         * \param[in] image The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] result The output OCR result will be writen to this structure.
         * \return true if the prediction successed, otherwise false.
         */
        virtual bool predict(cv::Mat* image, OCRResult* result);
        virtual bool predict(const cv::Mat& image, OCRResult* result);
        /** \brief BatchPredict the input image and get OCR result.
         *
         * \param[in] images The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] batch_result The output list of OCR result will be writen to this structure.
         * \return true if the prediction successed, otherwise false.
         */
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<OCRResult>* batch_result);

        bool initialized() const override;
        bool set_cls_batch_size(int cls_batch_size);
        int get_cls_batch_size();
        bool set_rec_batch_size(int rec_batch_size);
        int get_rec_batch_size();

    protected:
        std::unique_ptr<DBDetector> detector_ = nullptr;
        std::unique_ptr<Classifier> classifier_ = nullptr;
        std::unique_ptr<Recognizer> recognizer_ = nullptr;

    private:
        int cls_batch_size_ = 1;
        int rec_batch_size_ = 6;
    };
}
