// ModelDeploy demo_landmark：关键点扩展（车辆关键点 + 面部 Landmark 106 点）。
// Usage: demo_landmark <vehicle.onnx> <face.onnx> <image.jpg>
//   写第一个参数为 "none" 表示跳过车辆模型（无需对应权重）。
//   写第二个参数为 "none" 表示跳过人脸模型。
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/landmark/face_landmark.h"
#include "vision/landmark/vehicle_keypoint.h"

static void run_vehicle(const std::string& modelFile, const std::string& imgFile) {
    if (modelFile == "none") return;
    modeldeploy::RuntimeOption opt; opt.use_ort_backend();
    modeldeploy::vision::landmark::VehicleKeypoint model(modelFile, opt);
    if (!model.is_initialized()) { printf("vehicle init failed (missing weights?)\n"); return; }
    auto im = cv::imread(imgFile);
    if (im.empty()) { printf("cannot read image %s\n", imgFile.c_str()); return; }
    modeldeploy::vision::ImageData img(im);
    std::vector<modeldeploy::vision::KeyPointsResult> res;
    if (!model.predict(img, &res)) { printf("vehicle predict failed\n"); return; }
    printf("vehicle: %zu object(s), keypoints_num=%zu\n", res.size(),
           res.empty() ? 0 : res[0].keypoints.size());
    for (auto& r : res)
        printf("  box=(%.1f,%.1f,%.1f,%.1f) score=%.3f keypoints=%zu\n",
               r.box.x, r.box.y, r.box.width, r.box.height, r.score, r.keypoints.size());
}

static void run_face(const std::string& modelFile, const std::string& imgFile) {
    if (modelFile == "none") return;
    modeldeploy::RuntimeOption opt; opt.use_ort_backend();
    modeldeploy::vision::landmark::FaceLandmark model(modelFile, opt);
    if (!model.is_initialized()) { printf("face init failed (missing weights?)\n"); return; }
    auto im = cv::imread(imgFile);
    if (im.empty()) { printf("cannot read image %s\n", imgFile.c_str()); return; }
    modeldeploy::vision::ImageData img(im);
    std::vector<modeldeploy::vision::KeyPointsResult> res;
    if (!model.predict(img, &res)) { printf("face predict failed\n"); return; }
    printf("face: %zu result(s), keypoints_num=%zu\n", res.size(),
           res.empty() ? 0 : res[0].keypoints.size());
    for (auto& r : res)
        printf("  box=(%.1f,%.1f,%.1f,%.1f) keypoints=%zu\n",
               r.box.x, r.box.y, r.box.width, r.box.height, r.keypoints.size());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: demo_landmark <vehicle.onnx|none> <face.onnx|none> <image.jpg>\n");
        return 1;
    }
    run_vehicle(argv[1], argv[3]);
    run_face(argv[2], argv[3]);
    return 0;
}
