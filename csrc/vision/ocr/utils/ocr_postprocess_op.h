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
        void get_contour_area(const std::vector<std::vector<float>>& box,
                              float unclip_ratio, float& distance);

        cv::RotatedRect un_clip(std::vector<std::vector<float>> box,
                                const float& unclip_ratio);
        float** mat2_vec(cv::Mat mat);
        std::vector<std::vector<int>> order_points_clockwise(
            std::vector<std::vector<int>> pts);

        std::vector<std::vector<float>> get_mini_boxes(cv::RotatedRect box,
                                                       float& ssid);

        float box_score_fast(std::vector<std::vector<float>> box_array, cv::Mat pred);
        float polygon_score_acc(std::vector<cv::Point> contour, cv::Mat pred);

        std::vector<std::vector<std::vector<int>>> boxes_from_bitmap(
            cv::Mat pred, cv::Mat bitmap, const float& box_thresh,
            const float& det_db_unclip_ratio, const std::string& det_db_score_mode);

        std::vector<std::vector<std::vector<int>>> filter_tag_det_res(
            std::vector<std::vector<std::vector<int>>> boxes,
            const std::array<int, 4>& det_img_info);

    private:
        static bool x_sort_int(std::vector<int> a, std::vector<int> b);

        static bool x_sort_fp32(std::vector<float> a, std::vector<float> b);

        std::vector<std::vector<float>> mat2_vector(cv::Mat mat);

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
