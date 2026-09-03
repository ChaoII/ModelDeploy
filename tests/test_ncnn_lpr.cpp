//
// LPR（车牌检测 yolov5plate + 识别 plate_recognition_color）ncnn vs ORT 锚点测试。
// 补秩守卫：lpr_det 3D、lpr_rec 双输出 3D+2D（见 lpr_det/lpr_rec postprocessor.cpp）。
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "csrc/vision/lpr/lpr_models.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace lpr_test {

std::filesystem::path data_root() {
    const char* env = std::getenv("TEST_DATA_DIR");
    if (env && *env) return std::filesystem::path(env);
    return std::filesystem::current_path();
}
bool avail(const std::filesystem::path& p) { return std::filesystem::exists(p); }
RuntimeOption ort_opt() { return RuntimeOption(); }  // 默认 ORT
RuntimeOption ncnn_opt() {
    RuntimeOption o;
    o.use_ncnn_backend();
    o.set_device(Device::CPU, 0);
    return o;
}
std::filesystem::path lpr_model(const std::string& name) {
    return data_root() / "test_data" / "test_models" / "ncnn" / "lpr" / name / (name + ".param");
}
std::filesystem::path lpr_onnx(const std::string& name) {
    return data_root() / "test_data" / "test_models" / "onnx" / (name + ".onnx");
}
std::filesystem::path image(const std::string& name) {
    return data_root() / "test_data" / "test_images" / name;
}

// 目标框 IoU（Rect2f: x,y,width,height）。
float box_iou(const Rect2f& a, const Rect2f& b) {
    const float ax2 = a.x + a.width, ay2 = a.y + a.height;
    const float bx2 = b.x + b.width, by2 = b.y + b.height;
    const float ix = std::max(0.0f, std::min(ax2, bx2) - std::max(a.x, b.x));
    const float iy = std::max(0.0f, std::min(ay2, by2) - std::max(a.y, b.y));
    const float inter = ix * iy;
    const float ua = a.width * a.height + b.width * b.height - inter;
    return ua > 0.0f ? inter / ua : 0.0f;
}

// 返回分数最高的框；空则返回空 Rect2f。
Rect2f best_box(const std::vector<KeyPointsResult>& res) {
    if (res.empty()) return Rect2f();
    const auto it = std::max_element(res.begin(), res.end(),
        [](const KeyPointsResult& a, const KeyPointsResult& b) { return a.score < b.score; });
    return it->box;
}

}  // namespace lpr_test

#ifdef ENABLE_NCNN
TEST_CASE("LprDetection ncnn vs ORT", "[ncnn][phase_b][lpr][det]") {
    namespace p = lpr_test;
    const auto mdl = p::lpr_model("yolov5plate");
    const auto onx = p::lpr_onnx("yolov5plate");
    const auto imgf = p::image("test_lpr_pipeline.jpg");
    if (!p::avail(mdl) || !p::avail(imgf)) return;

    lpr::LprDetection ort(onx.string(), p::ort_opt());
    lpr::LprDetection ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;

    std::vector<KeyPointsResult> ro, rn;
    REQUIRE(ort.predict(img, &ro));
    REQUIRE(ncnn.predict(img, &rn));
    REQUIRE(!ro.empty());
    REQUIRE(!rn.empty());

    const Rect2f bo = p::best_box(ro), bn = p::best_box(rn);
    REQUIRE((bo.width > 0 && bo.height > 0));
    REQUIRE((bn.width > 0 && bn.height > 0));
    INFO("ort box=(" << bo.x << "," << bo.y << "," << bo.width << "," << bo.height
         << ") ncnn box=(" << bn.x << "," << bn.y << "," << bn.width << "," << bn.height << ")");
    REQUIRE(p::box_iou(bo, bn) > 0.5f);
}

TEST_CASE("LprRecognizer ncnn vs ORT", "[ncnn][phase_b][lpr][rec]") {
    namespace p = lpr_test;
    const auto mdl = p::lpr_model("plate_recognition_color");
    const auto onx = p::lpr_onnx("plate_recognition_color");
    const auto det_onx = p::lpr_onnx("yolov5plate");
    const auto imgf = p::image("test_lpr_pipeline.jpg");
    if (!p::avail(mdl) || !p::avail(imgf) || !p::avail(det_onx)) return;
    if (!p::avail(p::lpr_model("yolov5plate"))) return;

    // 用 ORT det 求车牌框，裁剪车牌区域作 rec 输入（双侧同输入，只对比引擎输出）。
    lpr::LprDetection det(det_onx.string(), p::ort_opt());
    REQUIRE(det.is_initialized());
    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;
    std::vector<KeyPointsResult> detres;
    REQUIRE(det.predict(img, &detres));
    if (detres.empty()) return;

    const Rect2f box = p::best_box(detres);
    if (box.width <= 0 || box.height <= 0) return;

    cv::Mat frame;
    REQUIRE(img.asMat(&frame));
    const int x0 = static_cast<int>(std::max(0.0f, box.x));
    const int y0 = static_cast<int>(std::max(0.0f, box.y));
    const int x1 = static_cast<int>(std::min<float>(frame.cols, box.x + box.width));
    const int y1 = static_cast<int>(std::min<float>(frame.rows, box.y + box.height));
    if (x1 - x0 <= 0 || y1 - y0 <= 0) return;
    cv::Mat crop = frame(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();
    ImageData crop_img(std::move(crop));

    lpr::LprRecognizer ort(onx.string(), p::ort_opt());
    lpr::LprRecognizer ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());

    LprResult r_ort, r_ncnn;
    REQUIRE(ort.predict(crop_img, &r_ort));
    REQUIRE(ncnn.predict(crop_img, &r_ncnn));
    INFO("ort str=[" << r_ort.car_plate_str << "] color=[" << r_ort.car_plate_color
         << "] ncnn str=[" << r_ncnn.car_plate_str << "] color=[" << r_ncnn.car_plate_color << "]");
    REQUIRE(!r_ort.car_plate_str.empty());
    REQUIRE(r_ncnn.car_plate_str == r_ort.car_plate_str);       // 车牌字符串强锚点
    REQUIRE(r_ncnn.car_plate_color == r_ort.car_plate_color);   // 颜色强锚点
}

TEST_CASE("LprPipeline ncnn vs ORT", "[ncnn][phase_b][lpr][pipe]") {
    namespace p = lpr_test;
    const auto dmdl = p::lpr_model("yolov5plate");
    const auto rmdl = p::lpr_model("plate_recognition_color");
    const auto imgf = p::image("test_lpr_pipeline.jpg");
    if (!p::avail(dmdl) || !p::avail(rmdl) || !p::avail(imgf)) return;

    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;
    // ORT：显式传 .onnx 防默认按扩展名误切 ncnn；ncnn：传 .param
    lpr::LprPipeline ort(p::lpr_onnx("yolov5plate").string(), p::lpr_onnx("plate_recognition_color").string(), p::ort_opt());
    lpr::LprPipeline ncnn(dmdl.string(), rmdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());

    std::vector<LprResult> ro, rn;
    REQUIRE(ort.predict(img, &ro));
    REQUIRE(ncnn.predict(img, &rn));
    REQUIRE((!ro.empty() && !rn.empty()));
    bool match = false;
    for (const auto& a : ro) for (const auto& b : rn)
        if (!a.car_plate_str.empty() && a.car_plate_str == b.car_plate_str && a.car_plate_color == b.car_plate_color) match = true;
    REQUIRE(match);  // 至少一个车牌字符串+颜色双侧一致
}
#endif  // ENABLE_NCNN
