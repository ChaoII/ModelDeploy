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
#include "qs_clarity.h"
#include "qs_no_mask.h"


#include <seeta/AgePredictor.h>
#include <seeta/GenderPredictor.h>
#include <seeta/EyeStateDetector.h>


using Status = seeta::FaceAntiSpoofing::Status;

class FaceModel {
public:
    explicit FaceModel(const std::string &model_dir, int thread_num);

    bool extract_feature(const SeetaImageData &image, std::vector<float> &feature);

    std::vector<SeetaFaceInfo> face_detection(const SeetaImageData &image);

    std::vector<SeetaPointF> face_marker(const SeetaImageData &image, const SeetaRect &rect);

    bool face_quality_authorize(const SeetaImageData &image);

    Status face_anti_spoofing(const SeetaImageData &image, const SeetaRect &rect, std::vector<SeetaPointF> points);


private:
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
    std::shared_ptr<seeta::QualityOfClarityEx> qs_clear_ = nullptr;
    std::shared_ptr<seeta::QualityOfNoMask> qs_no_mask_ = nullptr;
};


