//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <array>
#include "common/image_data.h"
#include "core/tensor.h"
#include "vision/common/result.h"
#include <opencv2/opencv.hpp>


namespace modeldeploy::vision::utils {
    MODELDEPLOY_CXX_EXPORT bool mat_to_tensor(cv::Mat& mat, Tensor* tensor, bool is_copy = true);

    bool image_data_to_tensor(const ImageData* image_data, Tensor* tensor);

    DataType cv_dtype_to_md_dtype(int type);

    cv::Point2f point2f_to_cv_type(Point2f point2f);

    cv::Point3f point3f_to_cv_type(Point3f point3f);

    cv::Rect2f rect2f_to_cv_type(Rect2f rect2f);

    cv::RotatedRect rotated_rect_to_cv_type(RotatedRect rotated_rect);


    MODELDEPLOY_CXX_EXPORT ImageData center_crop(const ImageData& image, const cv::Size& crop_size);

    bool mats_to_tensor(const std::vector<cv::Mat>& mats, Tensor* tensor);

    void obb_nms(std::vector<ObbResult>* result, float iou_threshold = 0.5, std::vector<int>* index = nullptr);

    void nms(std::vector<DetectionResult>* result, float iou_threshold = 0.5, std::vector<int>* index = nullptr);

    void nms(std::vector<InstanceSegResult>* result, float iou_threshold = 0.5, std::vector<int>* index = nullptr);

    void nms(std::vector<PoseResult>* result, float iou_threshold);

    void nms(std::vector<DetectionLandmarkResult>* result, float iou_threshold);

    void letter_box(cv::Mat* mat, const std::vector<int>& size, bool is_scale_up, bool is_mini_pad, bool is_no_pad,
                    const std::vector<float>& padding_value, int stride, LetterBoxRecord* letter_box_record);

    void print_mat_type(const cv::Mat& mat);

    std::vector<float> compute_sqrt(const std::vector<float>& vec);

    MODELDEPLOY_CXX_EXPORT std::vector<ImageData> align_face_with_five_points(
        const ImageData& image, std::vector<DetectionLandmarkResult>& result,
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

    std::array<float, 8> xcycwha_to_x1y1x2y2x3y3x4y4(float xc, float yc, float w, float h, float angle_rad);
    std::array<float, 5> x1y1x2y2x3y3x4y4_to_xcycwha(const std::array<float, 8>& pts);

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
