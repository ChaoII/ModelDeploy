//
// Created by AC on 2024-12-25.
//

#include "face_model.h"


using namespace seeta;


FaceModel::FaceModel(const std::string &model_dir, int thread_num) {

    ModelSetting fd_setting;
    fd_setting.append(model_dir + "/face_detector.csta");
    fd_setting.set_device(ModelSetting::CPU);
    detect_ = std::make_shared<FaceDetector>(fd_setting);
    detect_->set(seeta::FaceDetector::PROPERTY_NUMBER_THREADS, thread_num);
    // face_land_marker
    ModelSetting pd_setting;
    pd_setting.append(model_dir + "/face_landmarker_pts5.csta");
    pd_setting.set_device(ModelSetting::CPU);
    landmark_ = std::make_shared<FaceLandmarker>(pd_setting);
    //face_recognizer
    ModelSetting fr_setting;
    fr_setting.append(model_dir + "/face_recognizer.csta");
    fr_setting.set_device(ModelSetting::CPU);
    recognize_ = std::make_shared<FaceRecognizer>(fr_setting);
    recognize_->set(seeta::FaceRecognizer::PROPERTY_NUMBER_THREADS, thread_num);
    // face_anti_spoofing
    seeta::ModelSetting fs_setting;
    fs_setting.append(model_dir + "/fas_first.csta");
    fs_setting.append(model_dir + "/fas_second.csta");
    fs_setting.set_device(ModelSetting::CPU);
    anti_spoofing_ = std::make_shared<FaceAntiSpoofing>(fs_setting);
    anti_spoofing_->set(seeta::FaceAntiSpoofing::PROPERTY_NUMBER_THREADS, thread_num);
    anti_spoofing_->SetThreshold(0.3, 0.8);

    seeta::ModelSetting age_setting;
    age_setting.append(model_dir + "/age_predictor.csta");
    age_setting.set_device(ModelSetting::CPU);
    age_ = std::make_shared<seeta::AgePredictor>(age_setting);

    seeta::ModelSetting gender_setting;
    gender_setting.append(model_dir + "/gender_predictor.csta");
    gender_setting.set_device(ModelSetting::CPU);
    gender_ = std::make_shared<seeta::GenderPredictor>(gender_setting);

    seeta::ModelSetting eye_setting;
    eye_setting.append(model_dir + "/eye_state.csta");
    eye_setting.set_device(ModelSetting::CPU);
    eye_state_ = std::make_shared<seeta::EyeStateDetector>(eye_setting);

    qs_bright_ = std::make_shared<seeta::QualityOfBrightness>();
    qs_clarity_ = std::make_shared<seeta::QualityOfClarity>();
    qs_integrity_ = std::make_shared<seeta::QualityOfIntegrity>();
    qs_pose_ = std::make_shared<seeta::QualityOfPose>();
    qs_resolution_ = std::make_shared<seeta::QualityOfResolution>();
    qs_clear_ = std::make_shared<seeta::QualityOfClarityEx>(model_dir);
    qs_no_mask_ = std::make_shared<seeta::QualityOfNoMask>(landmark_);
}


bool FaceModel::extract_feature(const SeetaImageData &image, std::vector<float> &feature) {
    auto faces = detect_->detect(image);
    if (faces.size <= 0) {
        return false;
    }
    SeetaPointF points[5];
    landmark_->mark(image, faces.data[0].pos, points);
    recognize_->Extract(image, points, feature.data());
    return true;
}

std::vector<SeetaFaceInfo> FaceModel::face_detection(const SeetaImageData &image) {
    auto faces_ = detect_->detect(image);
    std::vector<SeetaFaceInfo> faces;
    for (int i = 0; i < faces_.size; i++) {
        faces.push_back(faces_.data[i]);
    }
    // 排序，将人脸由大到小进行排列
    std::partial_sort(faces.begin(), faces.begin() + 1, faces.end(),
                      [](SeetaFaceInfo a, SeetaFaceInfo b) {
                          return a.pos.width > b.pos.width;
                      });
    return faces;
}


std::vector<SeetaPointF> FaceModel::face_marker(const SeetaImageData &image, const SeetaRect &rect) {
    int point_nums = landmark_->number();
    std::vector<SeetaPointF> points(point_nums);
    landmark_->mark(image, rect, points.data());
    return points;
}

bool FaceModel::face_quality_authorize(const SeetaImageData &image) {
    shared_ptr<float> feature(new float[recognizer->GetExtractFeatureSize()]);
    bool rt = extractFeature(img, feature.get());
    if (!rt) {
        return false;
    }
    return true;
}

Status
FaceModel::face_anti_spoofing(const SeetaImageData &image, const SeetaRect &rect, std::vector<SeetaPointF> points) {
    return anti_spoofing_->Predict(image, rect, points.data());
}