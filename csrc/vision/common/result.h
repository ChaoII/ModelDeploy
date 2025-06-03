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

    enum class FaceAntiSpoofType:std::uint8_t {
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
        void display() const;
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
        [[nodiscard]] std::string str() const;
    };


    /*! @brief Face detection result structure for all the face detection models
     */
    // struct MODELDEPLOY_CXX_EXPORT DetectionLandmarkResult {
    //     /** \brief All the detected object boxes for an input image, the size of `boxes` is the number of detected objects, and the element of `boxes` is a array of 4 float values, means [xmin, ymin, xmax, ymax]
    //      */
    //     std::vector<cv::Rect2f> boxes;
    //     /** \brief
    //      * If the model detect face with landmarks, every detected object box correspoing to a landmark, which is a array of 2 float values, means location [x,y]
    //     */
    //     std::vector<cv::Point2f> landmarks;
    //
    //     std::vector<int> label_ids;
    //     /** \brief
    //      * Indicates the confidence of all targets detected from a single image, and the number of elements is consistent with boxes.size()
    //      */
    //     std::vector<float> scores;
    //     ResultType type = ResultType::FACE_DETECTION;
    //     /** \brief
    //      * `landmarks_per_face` indicates the number of face landmarks for each detected face
    //      * if the model's output contains face landmarks (such as YOLOv5Face, SCRFD, ...)
    //     */
    //     int landmarks_per_instance;
    //
    //     DetectionLandmarkResult() { landmarks_per_instance = 0; }
    //
    //     DetectionLandmarkResult(const DetectionLandmarkResult& res);
    //
    //     /// Clear FaceDetectionResult
    //     void clear();
    //
    //     /// Clear FaceDetectionResult and free the memory
    //     void free();
    //
    //     void reserve(size_t size);
    //
    //     void resize(size_t size);
    //
    //     /// Debug function, convert the result to string to print
    //     void display() const;
    // };

    struct MODELDEPLOY_CXX_EXPORT DetectionLandmarkResult {
        cv::Rect2f box;
        std::vector<cv::Point2f> landmarks;
        int label_id;
        float score;
        ResultType type = ResultType::FACE_DETECTION;
    };


    /*! @brief Face recognition result structure for all the Face recognition models
     */
    struct MODELDEPLOY_CXX_EXPORT FaceRecognitionResult {
        /** \brief The feature embedding that represents the final extraction of the face recognition model can be used to calculate the feature similarity between faces.
         */
        std::vector<float> embedding;

        ResultType type = ResultType::FACE_RECOGNITION;

        FaceRecognitionResult() = default;

        FaceRecognitionResult(const FaceRecognitionResult& res);

        /// Clear FaceRecognitionResult
        void clear();

        /// Clear FaceRecognitionResult and free the memory
        void free();

        void reserve(size_t size);

        void resize(size_t size);

        /// Debug function, convert the result to string to print
        void display();
    };

    struct MODELDEPLOY_CXX_EXPORT LprResult {
        cv::Rect2f box;
        std::vector<cv::Point2f> landmarks;
        int label_id;
        float score;
        std::string car_plate_str;
        std::string car_plate_color;
    };


    struct MODELDEPLOY_CXX_EXPORT FaceAntiSpoofResult {
        std::vector<FaceAntiSpoofType> anti_spoofs;

        FaceAntiSpoofResult() = default;

        FaceAntiSpoofResult(const FaceAntiSpoofResult& res);

        void clear();

        void free();

        void reserve(size_t size);

        void resize(size_t size);

        void display();
    };
}
