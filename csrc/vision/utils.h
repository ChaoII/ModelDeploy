//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <opencv2/opencv.hpp>
#include "csrc/core/tensor.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision::utils {
    bool mat_to_tensor(cv::Mat& mat, Tensor* tensor, bool is_copy = false);

    DataType cv_dtype_to_md_dtype(int type);

    MODELDEPLOY_CXX_EXPORT cv::Mat center_crop(const cv::Mat& image, const cv::Size& crop_size);

    bool mats_to_tensor(const std::vector<cv::Mat>& mats, Tensor* tensor);

    void nms(DetectionResult* output, float iou_threshold = 0.5, std::vector<int>* index = nullptr);

    void nms(DetectionLandmarkResult* result, float iou_threshold);

    void print_mat_type(const cv::Mat& mat);

    void sort_detection_result(DetectionResult* result);

    void sort_detection_result(DetectionLandmarkResult* result);

    std::vector<float> compute_sqrt(const std::vector<float>& vec);

    MODELDEPLOY_CXX_EXPORT std::vector<cv::Mat> align_face_with_five_points(
        const cv::Mat& image, DetectionLandmarkResult& result,
        std::vector<std::array<float, 2>> std_landmarks =
        {
            {89.3095f, 72.9025f},
            {169.3095f, 72.9025f},
            {127.8949f, 127.0441f},
            {96.8796f, 184.8907f},
            {159.1065f, 184.7601f},
        },
        std::array<int, 2> output_size = {256, 256});


    MODELDEPLOY_CXX_EXPORT float compute_similarity(const std::vector<float>& feature1,
                                                    const std::vector<float>& feature2);


    MODELDEPLOY_CXX_EXPORT std::vector<float> l2_normalize(const std::vector<float>& values);

    template <typename T>
    T clamp(T val, T min_val, T max_val) {
        return std::min(std::max(val, min_val), max_val);
    }


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
