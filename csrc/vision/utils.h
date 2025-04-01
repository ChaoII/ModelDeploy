//
// Created by aichao on 2025/2/20.
//

#pragma once
#include <opencv2/opencv.hpp>
#include "csrc/core/md_tensor.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision::utils {
    bool mat_to_tensor(cv::Mat& mat, MDTensor* tensor, bool is_copy = false);

    MDDataType::Type cv_dtype_to_md_dtype(int type);

    MODELDEPLOY_CXX_EXPORT cv::Mat center_crop(const cv::Mat& image, const cv::Size& crop_size);

    bool mats_to_tensor(const std::vector<cv::Mat>& mats, MDTensor* tensor);

    void nms(DetectionResult* output, float iou_threshold = 0.5, std::vector<int>* index = nullptr);

    void nms(DetectionLandmarkResult* result, float iou_threshold);

    void print_mat_type(const cv::Mat& mat);

    void sort_detection_result(DetectionResult* result);

    void sort_detection_result(DetectionLandmarkResult* result);

    std::vector<float> compute_sqrt(const std::vector<float>& vec);

    MODELDEPLOY_CXX_EXPORT std::vector<cv::Mat> AlignFaceWithFivePoints(
        cv::Mat& image, DetectionLandmarkResult& result,
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

    // cv::Mat transform_from_4points(cv::Mat &src_img, cv::Point2f  order_rect[4]) //透视变换
    // {
    //     cv::Point2f w1=order_rect[0]-order_rect[1];
    //     cv::Point2f w2=order_rect[2]-order_rect[3];
    //     auto width1 = getNorm2(w1.x,w1.y);
    //     auto width2 = getNorm2(w2.x,w2.y);
    //     auto maxWidth = std::max(width1,width2);
    //
    //     cv::Point2f h1=order_rect[0]-order_rect[3];
    //     cv::Point2f h2=order_rect[1]-order_rect[2];
    //     auto height1 = getNorm2(h1.x,h1.y);
    //     auto height2 = getNorm2(h2.x,h2.y);
    //     auto maxHeight = std::max(height1,height2);
    //     //  透视变换
    //     std::vector<cv::Point2f> pts_ori(4);
    //     std::vector<cv::Point2f> pts_std(4);
    //
    //     pts_ori[0]=order_rect[0];
    //     pts_ori[1]=order_rect[1];
    //     pts_ori[2]=order_rect[2];
    //     pts_ori[3]=order_rect[3];
    //
    //     pts_std[0]=cv::Point2f(0,0);
    //     pts_std[1]=cv::Point2f(maxWidth,0);
    //     pts_std[2]=cv::Point2f(maxWidth,maxHeight);
    //     pts_std[3]=cv::Point2f(0,maxHeight);
    //
    //     cv::Mat M = cv::getPerspectiveTransform(pts_ori,pts_std);
    //     cv:: Mat dstimg;
    //     cv::warpPerspective(src_img,dstimg,M,cv::Size(maxWidth,maxHeight));
    //     return dstimg;
    // }


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
