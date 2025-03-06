//
// Created by AC on 2024-12-25.
//

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>
#include <seeta/FaceAntiSpoofing.h>
#include <seeta/QualityOfBrightness.h>
#include <seeta/QualityOfClarity.h>
#include <seeta/QualityOfIntegrity.h>
#include <seeta/QualityOfPose.h>
#include <seeta/QualityOfResolution.h>
#include "csrc/vision/face/qs_clarity.h"
#include "csrc/vision/face/qs_no_mask.h"


#include <seeta/AgePredictor.h>
#include <seeta/GenderPredictor.h>
#include <seeta/EyeStateDetector.h>


using Status = seeta::FaceAntiSpoofing::Status;

class FaceModel {
public:
    enum class QualityEvaluateType : uint {
        BRIGHTNESS = 0,
        CLARITY = 1,
        INTEGRITY = 2,
        POSE = 3,
        RESOLUTION = 4,
        CLARITY_EX = 5,
        NO_MASK = 6
    };


    explicit FaceModel(const std::string& model_dir, int flag, int thread_num = 1);

    [[nodiscard]] std::vector<float>
    extract_feature(const SeetaImageData& image, std::vector<SeetaPointF> points) const;

    [[nodiscard]] std::vector<SeetaFaceInfo> face_detection(const SeetaImageData& image) const;

    [[nodiscard]] std::vector<SeetaPointF> face_marker(const SeetaImageData& image, const SeetaRect& rect) const;

    /// 人脸活体检测
    /// \param clarity_threshold 默认0.3
    /// \param reality_threshold 默认0.8
    void set_anti_spoofing_threshold(float clarity_threshold, float reality_threshold) const;

    [[nodiscard]] Status face_anti_spoofing(const SeetaImageData& image, const SeetaRect& rect,
                                            const std::vector<SeetaPointF>& points) const;

    [[nodiscard]] float face_feature_compare(const std::vector<float>& feature1,
                                             const std::vector<float>& feature2) const;

    [[nodiscard]] seeta::QualityResult quality_evaluate(const SeetaImageData& image, const SeetaRect& face,
                                                        const std::vector<SeetaPointF>& points,
                                                        QualityEvaluateType type) const;


    [[nodiscard]] int get_feature_size() const;

    [[nodiscard]] bool check_flag(int flag_check) const;

    [[nodiscard]] int age_predict(const SeetaImageData& image, const std::vector<SeetaPointF>& points) const;

    [[nodiscard]] seeta::GenderPredictor::GENDER gender_predict(const SeetaImageData& image,
                                                                const std::vector<SeetaPointF>& points) const;

    [[nodiscard]] std::pair<seeta::EyeStateDetector::EYE_STATE, seeta::EyeStateDetector::EYE_STATE>
    eye_state_predict(const SeetaImageData& img, const std::vector<SeetaPointF>& points) const;

private:
    int flag_;
    std::shared_ptr<seeta::FaceDetector> detect_ = nullptr;
    std::shared_ptr<seeta::FaceLandmarker> landmark_ = nullptr;
    std::shared_ptr<seeta::FaceRecognizer> recognize_ = nullptr;
    std::shared_ptr<seeta::FaceAntiSpoofing> anti_spoofing_ = nullptr;
    std::shared_ptr<seeta::AgePredictor> age_ = nullptr;
    std::shared_ptr<seeta::GenderPredictor> gender_ = nullptr;
    std::shared_ptr<seeta::EyeStateDetector> eye_state_ = nullptr;


    std::shared_ptr<seeta::QualityOfBrightness> qs_bright_ = nullptr;
    std::shared_ptr<seeta::QualityOfClarity> qs_clarity_ = nullptr;
    std::shared_ptr<seeta::QualityOfIntegrity> qs_integrity_ = nullptr;
    std::shared_ptr<seeta::QualityOfPose> qs_pose_ = nullptr;
    std::shared_ptr<seeta::QualityOfResolution> qs_resolution_ = nullptr;
    std::shared_ptr<seeta::QualityOfClarityEx> qs_clarity_ex_ = nullptr;
    std::shared_ptr<seeta::QualityOfNoMask> qs_no_mask_ = nullptr;
};
