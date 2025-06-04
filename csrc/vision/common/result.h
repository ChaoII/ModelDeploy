//
// Created by aichao on 2025/2/20.
//

#pragma once

#include "csrc/core/md_decl.h"
#include <opencv2/opencv.hpp>

namespace modeldeploy::vision {
    enum ResultType {
        CLASSIFY,
        DETECTION,
        OCR,
        FACE_DETECTION,
        FACE_RECOGNITION,
        MASK,
    };

    enum class FaceAntiSpoofResult:std::uint8_t {
        REAL,
        FUZZY,
        SPOOF,
    };


    /// Classify result structure for all the image classify models
    struct MODELDEPLOY_CXX_EXPORT ClassifyResult {
        ClassifyResult() = default;
        std::vector<int32_t> label_ids;
        std::vector<float> scores;
        std::vector<float> feature;
        ResultType type = ResultType::CLASSIFY;
        void reserve(int size);
        void resize(int size);
        void clear();
        void free();
        ClassifyResult(const ClassifyResult& other) = default;
        ClassifyResult& operator=(ClassifyResult&& other) noexcept;
    };

    /*! Mask structure, used in DetectionResult for instance segmentation models
     */
    struct MODELDEPLOY_CXX_EXPORT Mask {
        std::vector<uint8_t> buffer;
        std::vector<int64_t> shape; // (H,W) ...
        ResultType type = ResultType::MASK;
        void clear();
        void free();
        void* data() { return buffer.data(); }
        [[nodiscard]] const void* data() const { return buffer.data(); }
        void reserve(int size);
        void resize(int size);
        [[nodiscard]] std::string str() const;
    };

    /*! @brief Detection result structure for all the object detection models and instance segmentation models
     */
    struct MODELDEPLOY_CXX_EXPORT DetectionResult {
        cv::Rect2f box;
        int32_t label_id{};
        float score{};
        ResultType type = ResultType::DETECTION;
    };


    struct MODELDEPLOY_CXX_EXPORT InstanceSegResult {
        cv::Rect2f box;
        Mask mask;
        int32_t label_id{};
        float score{};
        ResultType type = ResultType::DETECTION;
    };

    struct MODELDEPLOY_CXX_EXPORT ObbResult {
        cv::RotatedRect rotated_box;
        int32_t label_id{};
        float score{};
        ResultType type = ResultType::DETECTION;
    };


    struct MODELDEPLOY_CXX_EXPORT PoseResult {
        cv::Rect2f box;
        std::vector<cv::Point3f> keypoints;
        int32_t label_id;
        float score;
        ResultType type = ResultType::FACE_DETECTION;
    };


    struct MODELDEPLOY_CXX_EXPORT OCRResult {
        std::vector<std::array<int, 8>> boxes;
        std::vector<std::string> text;
        std::vector<float> rec_scores;
        std::vector<float> cls_scores;
        std::vector<int32_t> cls_labels;
        std::vector<std::array<int, 8>> table_boxes;
        std::vector<std::string> table_structure;
        std::string table_html;
        ResultType type = ResultType::OCR;
        void clear();
    };


    struct MODELDEPLOY_CXX_EXPORT DetectionLandmarkResult {
        cv::Rect2f box;
        std::vector<cv::Point2f> landmarks;
        int label_id;
        float score;
        ResultType type = ResultType::FACE_DETECTION;
    };


    struct MODELDEPLOY_CXX_EXPORT FaceRecognitionResult {
        std::vector<float> embedding;
        ResultType type = ResultType::FACE_RECOGNITION;
        void display();
    };

    struct MODELDEPLOY_CXX_EXPORT LprResult {
        cv::Rect2f box;
        // 4 points
        std::vector<cv::Point2f> landmarks;
        int label_id;
        float score;
        std::string car_plate_str;
        std::string car_plate_color;
    };
}
