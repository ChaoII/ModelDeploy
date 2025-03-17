//
// Created by aichao on 2025/2/20.
//

#pragma once
#include <opencv2/opencv.hpp>
#include "../core/md_tensor.h"
#include "./common/result.h"

namespace modeldeploy::vision::utils {
    bool mat_to_tensor(cv::Mat& mat, MDTensor* tensor, bool is_copy = false);

    MDDataType::Type cv_dtype_to_md_dtype(int type);

    bool mats_to_tensor(const std::vector<cv::Mat>& mats, MDTensor* tensor);

    void nms(DetectionResult* output, float iou_threshold = 0.5, std::vector<int>* index = nullptr);

    void print_mat_type(const cv::Mat& mat);

    void sort_detection_result(DetectionResult* result);

    template <typename T>
    std::vector<int32_t> top_k_indices(const T* array, int array_size, int topk) {
        topk = std::min(array_size, topk);
        std::vector<int32_t> res(topk);
        std::set<int32_t> searched;
        for (int32_t i = 0; i < topk; ++i) {
            T min = static_cast<T>(-99999999);
            for (int32_t j = 0; j < array_size; ++j) {
                if (searched.find(j) != searched.end()) {
                    continue;
                }
                if (*(array + j) > min) {
                    res[i] = j;
                    min = *(array + j);
                }
            }
            searched.insert(res[i]);
        }
        return res;
    }
}
