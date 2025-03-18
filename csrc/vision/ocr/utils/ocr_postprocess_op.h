//
// Created by aichao on 2025/2/21.
//

#pragma once

#include <numeric>
#include <iomanip>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


namespace modeldeploy::vision::ocr {
    class PostProcessor {
    public:
        static void get_contour_area(const std::vector<std::vector<float>>& box,
                              float unclip_ratio, float& distance);

        static cv::RotatedRect un_clip(const std::vector<std::vector<float>>& box,
                                const float& unclip_ratio);
        static float** mat2_vec(cv::Mat mat);
        static std::vector<std::vector<int>> order_points_clockwise(
            std::vector<std::vector<int>> pts);

        static std::vector<std::vector<float>> get_mini_boxes(const cv::RotatedRect& box,
                                                       float& ssid);

        static float box_score_fast(std::vector<std::vector<float>> box_array, const cv::Mat& pred);
        static float polygon_score_acc(const std::vector<cv::Point>& contour, const cv::Mat& pred);

        static std::vector<std::vector<std::vector<int>>> boxes_from_bitmap(
            const cv::Mat& pred, const cv::Mat& bitmap, const float& box_thresh,
            const float& det_db_unclip_ratio, const std::string& det_db_score_mode);

        static std::vector<std::vector<std::vector<int>>> filter_tag_det_res(
            std::vector<std::vector<std::vector<int>>> boxes,
            const std::array<int, 4>& det_img_info);

    private:
        static bool x_sort_int(const std::vector<int>& a, const std::vector<int>& b);

        static bool x_sort_fp32(const std::vector<float>& a, const std::vector<float>& b);

        static std::vector<std::vector<float>> mat2_vector(cv::Mat mat);

        static int _max(const int a, const int b) { return a >= b ? a : b; }

        static int _min(const int a, const int b) { return a >= b ? b : a; }

        template <class T>
        static T clamp(T x, T min, T max) {
            if (x > max) return max;
            if (x < min) return min;
            return x;
        }
    };
} // namespace ocr
