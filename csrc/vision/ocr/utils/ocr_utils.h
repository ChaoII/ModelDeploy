//
// Created by aichao on 2025/2/21.
//
#pragma once

#include <vector>

#include "opencv2/core.hpp"

namespace modeldeploy::vision::ocr {
    cv::Mat get_rotate_crop_image(const cv::Mat& image,
                                  const std::array<int, 8>& box);

    void sort_boxes(std::vector<std::array<int, 8>>* boxes);

    std::vector<int> arg_sort(const std::vector<float>& array);

    std::vector<float> softmax(std::vector<float>& src);

    std::vector<int> xyxyxyxy2xyxy(const std::array<int, 8>& box);

    float dis(const std::vector<int>& box1, const std::vector<int>& box2);

    float iou(const std::vector<int>& box1, const std::vector<int>& box2);

    bool comparison_dis(const std::vector<float>& dis1,
                        const std::vector<float>& dis2);
}
