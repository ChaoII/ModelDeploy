//
// Created by AC on 2024-12-25.
//

#include <filesystem>
#include "capi/common/md_micro.h"
#include "csrc/vision/face/face_model.h"


using namespace seeta;
namespace fs = std::filesystem;

FaceModel::FaceModel(const std::string& model_dir, int flag, int thread_num) {
    flag_ = flag;
    if (flag_ & MD_FACE_DETECT) {
        ModelSetting fd_setting;
        auto model_path = fs::path(model_dir) / "face_detector.csta";
        fd_setting.append(model_path.string());
        fd_setting.set_device(ModelSetting::CPU);
        detect_ = std::make_shared<FaceDetector>(fd_setting);
        detect_->set(seeta::FaceDetector::PROPERTY_NUMBER_THREADS, thread_num);
    }
    if (flag_ & MD_FACE_LANDMARK) {
        // face_land_marker
        ModelSetting pd_setting;
        auto model_path = fs::path(model_dir) / fs::path("face_landmarker_pts5.csta");
        pd_setting.append(model_path.string());
        pd_setting.set_device(ModelSetting::CPU);
        landmark_ = std::make_shared<FaceLandmarker>(pd_setting);
    }

    if (flag_ & MD_FACE_RECOGNITION) {
        //face_recognizer
        ModelSetting fr_setting;
        auto model_path = fs::path(model_dir) / fs::path("face_recognizer.csta");
        fr_setting.append(model_path.string());
        fr_setting.set_device(ModelSetting::CPU);
        recognize_ = std::make_shared<FaceRecognizer>(fr_setting);
        recognize_->set(seeta::FaceRecognizer::PROPERTY_NUMBER_THREADS, thread_num);
    }

    if (flag_ & MD_FACE_ANTI_SPOOfING) {
        // face_anti_spoofing
        seeta::ModelSetting fs_setting;
        auto model_path_first = fs::path(model_dir) / fs::path("fas_first.csta");
        auto model_path_second = fs::path(model_dir) / fs::path("fas_second.csta");
        fs_setting.append(model_path_first.string());
        fs_setting.append(model_path_second.string());
        fs_setting.set_device(ModelSetting::CPU);
        anti_spoofing_ = std::make_shared<FaceAntiSpoofing>(fs_setting);
        anti_spoofing_->set(seeta::FaceAntiSpoofing::PROPERTY_NUMBER_THREADS, thread_num);
        anti_spoofing_->SetThreshold(0.3, 0.8);
    }

    if (flag_ & MD_FACE_QUALITY_EVALUATE) {
        qs_bright_ = std::make_shared<seeta::QualityOfBrightness>();
        qs_clarity_ = std::make_shared<seeta::QualityOfClarity>();
        qs_integrity_ = std::make_shared<seeta::QualityOfIntegrity>();
        qs_pose_ = std::make_shared<seeta::QualityOfPose>();
        qs_resolution_ = std::make_shared<seeta::QualityOfResolution>();
        qs_clarity_ex_ = std::make_shared<seeta::QualityOfClarityEx>(model_dir);
        qs_no_mask_ = std::make_shared<seeta::QualityOfNoMask>(landmark_);
    }

    if (flag_ & MD_FACE_AGE_ATTRIBUTE) {
        seeta::ModelSetting age_setting;
        auto model_path = fs::path(model_dir) / "age_predictor.csta";
        age_setting.append(model_path.string());
        age_setting.set_device(ModelSetting::CPU);
        age_ = std::make_shared<seeta::AgePredictor>(age_setting);
    }

    if (flag_ & MD_FACE_GENDER_ATTRIBUTE) {
        seeta::ModelSetting gender_setting;
        auto model_path = fs::path(model_dir) / "gender_predictor.csta";
        gender_setting.append(model_path.string());
        gender_setting.set_device(ModelSetting::CPU);
        gender_ = std::make_shared<seeta::GenderPredictor>(gender_setting);
    }

    if (flag_ & MD_FACE_EYE_STATE) {
        seeta::ModelSetting eye_setting;
        auto model_path = fs::path(model_dir) / "eye_state.csta";
        eye_setting.append(model_path.string());
        eye_setting.set_device(ModelSetting::CPU);
        eye_state_ = std::make_shared<seeta::EyeStateDetector>(eye_setting);
    }
}


std::vector<float> FaceModel::extract_feature(const SeetaImageData& image, std::vector<SeetaPointF> points) const {
    if (!(flag_ & MD_FACE_RECOGNITION)) throw std::runtime_error("FACE_RECOGNITION flag is not enabled");
    assert(points.size() == 5);
    std::vector<float> feature(recognize_->GetExtractFeatureSize());
    recognize_->Extract(image, points.data(), feature.data());
    return feature;
}

int FaceModel::get_feature_size() const {
    if (!(flag_ & MD_FACE_RECOGNITION)) throw std::runtime_error("FACE_RECOGNITION flag is not enabled");
    return recognize_->GetExtractFeatureSize();
}

float FaceModel::face_feature_compare(const std::vector<float>& feature1, const std::vector<float>& feature2) const {
    if (!(flag_ & MD_FACE_RECOGNITION)) throw std::runtime_error("MD_FACE_RECOGNITION flag is not enabled");
    return recognize_->CalculateSimilarity(feature1.data(), feature2.data());
}

std::vector<SeetaFaceInfo> FaceModel::face_detection(const SeetaImageData& image) const {
    if (!(flag_ & MD_FACE_DETECT))throw std::runtime_error("FACE_DETECT flag is not enabled");
    const auto faces_ = detect_->detect(image);
    std::vector<SeetaFaceInfo> faces;
    faces.reserve(faces_.size);
    for (int i = 0; i < faces_.size; i++) {
        faces.emplace_back(faces_.data[i]);
    }
    // 重新排序
    std::partial_sort(faces.begin(), faces.begin() + 1, faces.end(),
                      [](const SeetaFaceInfo& a, const SeetaFaceInfo& b) {
                          return a.pos.width > b.pos.width;
                      });
    return faces;
}


std::vector<SeetaPointF> FaceModel::face_marker(const SeetaImageData& image, const SeetaRect& rect) const {
    if (!(flag_ & MD_FACE_LANDMARK)) throw std::runtime_error("FACE_LANDMARK flag is not enabled");
    const int point_nums = landmark_->number();
    std::vector<SeetaPointF> points(point_nums);
    landmark_->mark(image, rect, points.data());
    return points;
}

void FaceModel::set_anti_spoofing_threshold(const float clarity_threshold, const float reality_threshold) const {
    if (!(flag_ & MD_FACE_ANTI_SPOOfING)) throw std::runtime_error("FACE_ANTI_SPOOfING flag is not enabled");
    anti_spoofing_->SetThreshold(clarity_threshold, reality_threshold);
}

Status FaceModel::face_anti_spoofing(const SeetaImageData& image, const SeetaRect& rect,
                                     const std::vector<SeetaPointF>& points) const {
    if (!(flag_ & MD_FACE_ANTI_SPOOfING)) throw std::runtime_error("FACE_ANTI_SPOOfING flag is not enabled");
    return anti_spoofing_->Predict(image, rect, points.data());
}


seeta::QualityResult FaceModel::quality_evaluate(const SeetaImageData& image,
                                                 const SeetaRect& face, const std::vector<SeetaPointF>& points,
                                                 const QualityEvaluateType type) const {
    if (!(flag_ & MD_FACE_QUALITY_EVALUATE)) {
        throw std::runtime_error("FACE_QUALITY_EVALUATE flag is not enabled");
    }
    switch (type) {
    case QualityEvaluateType::BRIGHTNESS:
        return qs_bright_->check(image, face, points.data(), static_cast<int>(points.size()));
    case QualityEvaluateType::CLARITY:
        return qs_clarity_->check(image, face, points.data(), static_cast<int>(points.size()));
    case QualityEvaluateType::INTEGRITY:
        return qs_integrity_->check(image, face, points.data(), static_cast<int>(points.size()));
    case QualityEvaluateType::POSE:
        return qs_pose_->check(image, face, points.data(), static_cast<int>(points.size()));
    case QualityEvaluateType::RESOLUTION:
        return qs_resolution_->check(image, face, points.data(), static_cast<int>(points.size()));
    case QualityEvaluateType::CLARITY_EX:
        return qs_clarity_ex_->check(image, face, points.data(), static_cast<int>(points.size()));
    case QualityEvaluateType::NO_MASK:
        return qs_no_mask_->check(image, face, points.data(), static_cast<int>(points.size()));
    default:
        throw std::runtime_error("QualityEvaluateType is not supported");
    }
}

bool FaceModel::check_flag(int flag_check) const {
    return flag_ & flag_check;
}

int FaceModel::age_predict(const SeetaImageData& image, const std::vector<SeetaPointF>& points) const {
    if (!(flag_ & MD_FACE_AGE_ATTRIBUTE)) throw std::runtime_error("FACE_AGE_ATTRIBUTE flag is not enabled");
    assert(points.size() == 5);
    int age = 0;
    age_->PredictAgeWithCrop(image, points.data(), age);
    return age;
}

seeta::GenderPredictor::GENDER FaceModel::gender_predict(const SeetaImageData& image,
                                                         const std::vector<SeetaPointF>& points) const {
    if (!(flag_ & MD_FACE_GENDER_ATTRIBUTE)) throw std::runtime_error("FACE_GENDER_ATTRIBUTE flag is not enabled");
    assert(points.size() == 5);
    seeta::GenderPredictor::GENDER gender = seeta::GenderPredictor::GENDER::MALE;
    gender_->PredictGenderWithCrop(image, points.data(), gender);
    return gender;
}

std::pair<seeta::EyeStateDetector::EYE_STATE, seeta::EyeStateDetector::EYE_STATE>
FaceModel::eye_state_predict(const SeetaImageData& img,
                             const std::vector<SeetaPointF>& points) const {
    if (!(flag_ & MD_FACE_EYE_STATE)) throw std::runtime_error("FACE_EYE_STATE flag is not enabled");
    seeta::EyeStateDetector::EYE_STATE left_eye, right_eye;
    eye_state_->Detect(img, points.data(), left_eye, right_eye);
    return std::make_pair(left_eye, right_eye);
}
