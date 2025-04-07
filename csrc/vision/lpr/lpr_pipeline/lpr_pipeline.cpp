//
// Created by aichao on 2025/2/21.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/lpr/lpr_pipeline/lpr_pipeline.h"

namespace modeldeploy::vision::lpr {
    LprPipeline::LprPipeline(const std::string& det_model_path,
                             const std::string& rec_model_path,
                             int thread_num) {
        RuntimeOption option;
        option.set_cpu_thread_num(thread_num);
        detector_ = std::make_unique<LprDetection>(det_model_path, option);
        recognizer_ = std::make_unique<LprRecognizer>(rec_model_path, option);
    }

    LprPipeline::~LprPipeline() = default;

    bool LprPipeline::is_initialized() const {
        if (detector_ != nullptr && !detector_->is_initialized()) {
            return false;
        }
        if (recognizer_ != nullptr && !recognizer_->is_initialized()) {
            return false;
        }
        return true;
    }


    float get_norm2(const float x, const float y) {
        return sqrt(x * x + y * y);
    }

    cv::Mat transform_from_4points(const cv::Mat& src_image, const std::array<cv::Point2f, 4>& order_rect) //透视变换
    {
        const cv::Point2f w1 = order_rect[0] - order_rect[1];
        const cv::Point2f w2 = order_rect[2] - order_rect[3];
        const auto width1 = get_norm2(w1.x, w1.y);
        const auto width2 = get_norm2(w2.x, w2.y);
        const auto max_width = std::max(width1, width2);

        const cv::Point2f h1 = order_rect[0] - order_rect[3];
        const cv::Point2f h2 = order_rect[1] - order_rect[2];
        const auto height1 = get_norm2(h1.x, h1.y);
        const auto height2 = get_norm2(h2.x, h2.y);
        const auto max_height = std::max(height1, height2);
        //  透视变换
        std::vector<cv::Point2f> pts_ori(4);
        std::vector<cv::Point2f> pts_std(4);

        pts_ori[0] = order_rect[0];
        pts_ori[1] = order_rect[1];
        pts_ori[2] = order_rect[2];
        pts_ori[3] = order_rect[3];

        pts_std[0] = cv::Point2f(0, 0);
        pts_std[1] = cv::Point2f(max_width, 0);
        pts_std[2] = cv::Point2f(max_width, max_height);
        pts_std[3] = cv::Point2f(0, max_height);

        const cv::Mat M = cv::getPerspectiveTransform(pts_ori, pts_std);
        cv::Mat dst_image;
        cv::warpPerspective(src_image, dst_image, M, cv::Size2f(max_width, max_height));
        return dst_image;
    }

    // 双层车牌 分割 拼接
    cv::Mat get_split_merge(const cv::Mat& img) {
        const auto upper_rect_area = cv::Rect(0, 0, img.cols,
                                              static_cast<int>(5.0 / 12 * img.rows));
        const auto lower_rect_area = cv::Rect(0, static_cast<int>(1.0 / 3 * img.rows), img.cols,
                                              img.rows - static_cast<int>(1.0 / 3 * img.rows));
        cv::Mat img_upper = img(upper_rect_area);
        const cv::Mat img_lower = img(lower_rect_area);
        cv::resize(img_upper, img_upper, img_lower.size());
        cv::Mat out(img_lower.rows, img_lower.cols + img_upper.cols, CV_8UC3, cv::Scalar(114, 114, 114));
        img_upper.copyTo(out(cv::Rect(0, 0, img_upper.cols, img_upper.rows)));
        img_lower.copyTo(out(cv::Rect(img_upper.cols, 0, img_lower.cols, img_lower.rows)));
        return out;
    }

    bool LprPipeline::predict(const cv::Mat& image, LprResult* results) {
        DetectionLandmarkResult det_result;
        if (!detector_->predict(image, &det_result)) {
            MD_LOG_ERROR << "detector predict failed" << std::endl;
            return false;
        }
        const size_t lp_num = det_result.boxes.size();
        results->resize(lp_num);
        results->landmarks = det_result.landmarks;
        results->boxes = det_result.boxes;
        results->scores = det_result.scores;
        results->label_ids = det_result.label_ids;
        for (int i = 0; i < lp_num; ++i) {
            std::array<cv::Point2f, 4> points;
            points[0] = cv::Point2f(det_result.landmarks[i * 4 + 0][0], det_result.landmarks[i * 4 + 0][1]);
            points[1] = cv::Point2f(det_result.landmarks[i * 4 + 1][0], det_result.landmarks[i * 4 + 1][1]);
            points[2] = cv::Point2f(det_result.landmarks[i * 4 + 2][0], det_result.landmarks[i * 4 + 2][1]);
            points[3] = cv::Point2f(det_result.landmarks[i * 4 + 3][0], det_result.landmarks[i * 4 + 3][1]);
            cv::Mat transform_image = transform_from_4points(image, points);
            // 如果是双层车牌
            if (det_result.label_ids[i]) {
                transform_image = get_split_merge(transform_image);
            }
            LprResult tmp_result;
            recognizer_->predict(transform_image, &tmp_result);
            results->car_plate_colors[i] = tmp_result.car_plate_colors[0];
            results->car_plate_strs[i] = tmp_result.car_plate_strs[0];
        }
        return true;
    }
}
