//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <vector>
#include <array>
#include "core/md_decl.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision {
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
        ClassifyResult(const ClassifyResult& other) = default;
        ClassifyResult& operator=(ClassifyResult&& other) noexcept = default;
    };

    /*! Mask structure, used in DetectionResult for instance segmentation models
     */
    struct MODELDEPLOY_CXX_EXPORT Mask {
        std::vector<uint8_t> buffer;
        std::vector<int64_t> shape; // (H,W) ...
        void* data() { return buffer.data(); }
        [[nodiscard]] const void* data() const { return buffer.data(); }
        void resize(int size);
    };

    /*! @brief Detection result structure for all the object detection models and instance segmentation models
     */
    struct MODELDEPLOY_CXX_EXPORT DetectionResult {
        Rect2f box;
        int32_t label_id{};
        float score{};
    };


    struct MODELDEPLOY_CXX_EXPORT InstanceSegResult {
        Rect2f box;
        Mask mask;
        int32_t label_id{};
        float score{};
    };

    /*! @brief Semantic segmentation result structure for yolo26n-sem (cityscapes 19 classes)
     * 每像素类别索引 [0, num_classes)，shape 为 (H, W)
     */
    struct MODELDEPLOY_CXX_EXPORT SemSegResult {
        std::vector<uint8_t> labels;
        std::vector<int64_t> shape; // (H, W)
        int32_t num_classes{};
    };

    /*! @brief Depth estimation result structure for yolo26n-depth (log-depth 模型输出经 exp 还原)
     * 每像素深度值（米），shape 为 (H, W)
     */
    struct MODELDEPLOY_CXX_EXPORT DepthResult {
        std::vector<float> depth;
        std::vector<int64_t> shape; // (H, W)
    };

    struct MODELDEPLOY_CXX_EXPORT ObbResult {
        RotatedRect rotated_box;
        int32_t label_id{};
        float score{};
    };


    struct MODELDEPLOY_CXX_EXPORT KeyPointsResult {
        Rect2f box;
        std::vector<Point3f> keypoints;
        int32_t label_id{};
        float score{};
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
    };

    struct MODELDEPLOY_CXX_EXPORT FaceRecognitionResult {
        std::vector<float> embedding;
    };

    /*! @brief Pedestrian Re-ID result structure
     *  OSNet 输出经 L2 归一化后的 512-d 特征向量
     */
    struct MODELDEPLOY_CXX_EXPORT ReIdResult {
        std::vector<float> embedding;  //!< 已 L2 归一化的 512-d 特征
    };

    struct MODELDEPLOY_CXX_EXPORT LprResult {
        Rect2f box;
        // 4 points
        std::vector<Point3f> keypoints;
        int label_id{};
        float score{};
        std::string car_plate_str;
        std::string car_plate_color;
    };

    struct MODELDEPLOY_CXX_EXPORT AttributeResult {
        Rect2f box;
        int32_t box_label_id{};
        float box_score{};
        std::vector<float> attr_scores{};
    };
}
