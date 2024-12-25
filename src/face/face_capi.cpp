//
// Created by AC on 2024-12-25.
//

#include "face_capi.h"


#include <opencv2/opencv.hpp>
#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>
#include <seeta/FaceTracker.h>
#include <seeta/FaceAntiSpoofing.h>


using namespace std;
using namespace seeta;

MDStatusCode md_create_face_model(MDModel *model, const char *model_dir, int thread_num = 8) {

    ModelSetting fd_setting;
    fd_setting.append(model_dir + "face_detector.csta");
    fd_setting.set_device(ModelSetting::CPU);
    FD_ = std::make_shared<FaceDetector>(fd_setting);
    FD_->set(seeta::FaceDetector::PROPERTY_NUMBER_THREADS, face_recognition_thread_num);
    // face_land_marker
    ModelSetting pd_setting;
    pd_setting.append(model_dir + "face_landmarker_pts5.csta");
    pd_setting.set_device(ModelSetting::CPU);
    FL_ = std::make_shared<FaceLandmarker>(pd_setting);
    //face_recognizer
    ModelSetting fr_setting;
    fr_setting.append(model_dir + "face_recognizer.csta");
    fr_setting.set_device(ModelSetting::CPU);
    FR_ = std::make_shared<FaceRecognizer>(fr_setting);
    FR_->set(seeta::FaceRecognizer::PROPERTY_NUMBER_THREADS, face_recognition_thread_num);
    // face_anti_spoofing
    seeta::ModelSetting fs_setting;
    fs_setting.append(model_dir + "fas_first.csta");
    fs_setting.set_device(ModelSetting::CPU);
    FS_ = std::make_shared<FaceAntiSpoofing>(fs_setting);
    FS_->set(seeta::FaceAntiSpoofing::PROPERTY_NUMBER_THREADS, face_recognition_thread_num);
    // unsupported set thread number

}



bool SeetaFace::extractFeature(cv::Mat &img, float *feature) {

    SeetaImageData data = utils::cvMatToSImg(img);
    auto faces = FD_->detect(data);
    if (faces.size <= 0) {
        return false;
    }
    SeetaPointF points[5];
    FL_->mark(data, faces.data[0].pos, points);
    FR_->Extract(data, points, feature);
    return true;
}

std::vector<SeetaFaceInfo> SeetaFace::faceDetection(cv::Mat &img) {
    SeetaImageData data = utils::cvMatToSImg(img);
    auto faces_ = FD_->detect(data);
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

QPair<int64_t, float> SeetaFace::faceRecognition(cv::Mat &img, std::vector<SeetaPointF> points) {
    SeetaImageData data = utils::cvMatToSImg(img);
    unique_ptr<float[]> feature(new float[FR_->GetExtractFeatureSize()]);
    FR_->Extract(data, points.data(), feature.get());
    SearchResult result = VectorSearch::getInstance().search(feature.get(), 3);
    return {result.I[0], result.D[0]};
}

std::vector<SeetaPointF> SeetaFace::faceMarker(cv::Mat &img, const SeetaRect &rect) {
    SeetaImageData data = utils::cvMatToSImg(img);
    int point_nums = FL_->number();
    std::vector<SeetaPointF> points(point_nums);
    FL_->mark(data, rect, points.data());
    return points;
}

bool SeetaFace::faceQualityAuthorize(cv::Mat &img) {
    shared_ptr<float> feature(new float[FR_->GetExtractFeatureSize()]);
    bool rt = extractFeature(img, feature.get());
    if (!rt) {
        return false;
    }
    return true;
}

Status SeetaFace::faceAntiSpoofing(cv::Mat &img, const SeetaRect &rect, std::vector<SeetaPointF> points) {
    SeetaImageData data = utils::cvMatToSImg(img);
    return FS_->Predict(data, rect, points.data());
}







