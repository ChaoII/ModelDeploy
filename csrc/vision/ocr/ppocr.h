//
// Created by aichao on 2025/2/21.
//

#pragma once

#include <vector>
#include "csrc/core/md_decl.h"
#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/ocr/classifier.h"
#include "csrc/vision/ocr/dbdetector.h"
#include "csrc/vision/ocr/recognizer.h"
#include "csrc/vision/ocr/utils/ocr_postprocess_op.h"

namespace modeldeploy::vision::ocr {
    class MODELDEPLOY_CXX_EXPORT PaddleOCR : public BaseModel {
    public:
        PaddleOCR(const std::string& det_model_path,
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

        ~PaddleOCR() override;


        [[nodiscard]] std::string name() const override { return "PaddleOCR"; }

        /** \brief Predict the input image and get OCR result.
         *
         * \param[in] image The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] result The output OCR result will be writen to this structure.
         * \param timers
         * \return true if the prediction successed, otherwise false.
         */
        virtual bool predict(cv::Mat* image, OCRResult* result, TimerArray* timers = nullptr);

        virtual bool predict(const cv::Mat& image, OCRResult* result, TimerArray* timers = nullptr);

        /** \brief BatchPredict the input image and get OCR result.
         *
         * \param[in] images The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] batch_result The output list of OCR result will be writen to this structure.
         * \param timers
         * \return true if the prediction successed, otherwise false.
         */
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<OCRResult>* batch_result, TimerArray* timers = nullptr);

        [[nodiscard]] bool is_initialized() const override;

        bool set_cls_batch_size(int cls_batch_size);

        [[nodiscard]] int get_cls_batch_size() const;

        bool set_rec_batch_size(int rec_batch_size);

        [[nodiscard]] int get_rec_batch_size() const;

    protected:
        std::unique_ptr<DBDetector> detector_ = nullptr;
        std::unique_ptr<Classifier> classifier_ = nullptr;
        std::unique_ptr<Recognizer> recognizer_ = nullptr;

    private:
        int cls_batch_size_ = 1;
        int rec_batch_size_ = 6;
    };
}
