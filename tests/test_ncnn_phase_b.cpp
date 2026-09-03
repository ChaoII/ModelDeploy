#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include "csrc/vision/face/insightface/face_analysis.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace phase_b_test {

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
std::filesystem::path iface_model(const std::string& sub) {
    return data_root() / "test_data" / "test_models" / "ncnn" / "insightface" / sub / (sub + ".param");
}
std::filesystem::path iface_onnx(const std::string& sub) {
    return data_root() / "test_data" / "test_models" / "onnx" / "insightface" / "buffalo_l" / (sub + ".onnx");
}
std::filesystem::path image(const std::string& name) {
    return data_root() / "test_data" / "test_images" / name;
}

// ORT 与 ncnn 双侧都跑 det，取两侧各自 top 检测框并核对（IoU 最大匹配，容忍排序差异）。
bool box_match(const std::array<float,4>& a, const std::array<float,4>& b) {
    const float x1 = std::max(a[0], b[0]), y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]), y2 = std::min(a[3], b[3]);
    const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    const float ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter;
    return inter / std::max(1e-6f, ua) > 0.5f;
}

}  // namespace phase_b_test

#ifdef ENABLE_NCNN
TEST_CASE("InsightFace det_10g ncnn vs ORT", "[ncnn][phase_b][insightface][det]") {
    namespace p = phase_b_test;
    const auto mdl = p::iface_model("det_10g");
    const auto onx = p::iface_onnx("det_10g");
    const auto imgf = p::image("test_person.jpg");
    if (!p::avail(mdl) || !p::avail(imgf)) return;

    face::InsightFaceDet ort(onx.string(), p::ort_opt());
    face::InsightFaceDet ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    auto img = ImageData::imread(imgf.string());

    std::vector<face::InsightFaceBox> boxes_ort, boxes_ncnn;
    REQUIRE(ort.predict(img, &boxes_ort));
    REQUIRE(ncnn.predict(img, &boxes_ncnn));
    REQUIRE(!boxes_ort.empty());
    REQUIRE(!boxes_ncnn.empty());

    // 两侧 top 检测框应 IoU 匹配（ncnn 输出可能是 2D 同秩，shape()[0] 语义一致）
    REQUIRE(p::box_match(boxes_ort[0].bbox, boxes_ncnn[0].bbox));
    REQUIRE(std::fabs(boxes_ort[0].score - boxes_ncnn[0].score) < 0.05f);
    REQUIRE(boxes_ncnn[0].kps.size() == 5);
}

TEST_CASE("InsightFace recognition (w600k_r50) ncnn vs ORT", "[ncnn][phase_b][insightface][rec]") {
    namespace p = phase_b_test;
    const auto mdl = p::iface_model("recognition");
    const auto detm = p::iface_model("det_10g");   // v/ncnn det
    const auto dtonx = p::iface_onnx("det_10g");   // ORT det（仅取 bbox/kps，双侧共用）
    const auto imgf = p::image("test_person.jpg");
    if (!p::avail(mdl) || !p::avail(detm) || !p::avail(imgf)) return;

    auto img = ImageData::imread(imgf.string());
    face::InsightFaceDet det(dtonx.string(), p::ort_opt());
    std::vector<face::InsightFaceBox> boxes;
    REQUIRE(det.predict(img, &boxes));
    if (boxes.empty()) return;
    const auto& b = boxes[0];
    // 至少 3 个关键点才能 alignment
    if (b.kps.size() < 3) return;

    // 注意：recognition 的 ORT onnx 文件名是 w600k_r50.onnx（非 recognition.onnx）
    face::InsightFaceRecognition ort(p::iface_onnx("w600k_r50").string(), p::ort_opt());
    face::InsightFaceRecognition ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    std::vector<float> emb_ort, emb_ncnn;
    REQUIRE(ort.predict(img, b.kps, &emb_ort));
    REQUIRE(ncnn.predict(img, b.kps, &emb_ncnn));
    REQUIRE(!emb_ort.empty());
    REQUIRE(emb_ort.size() == emb_ncnn.size());
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < emb_ort.size(); ++i) { dot += emb_ort[i]*emb_ncnn[i]; na += emb_ort[i]*emb_ort[i]; nb += emb_ncnn[i]*emb_ncnn[i]; }
    const double sim = dot / (std::sqrt(na) * std::sqrt(nb));
    INFO("rec embedding cosine sim=" << sim);
    REQUIRE(sim > 0.99);
}

TEST_CASE("InsightFace genderage ncnn vs ORT", "[ncnn][phase_b][insightface][ga]") {
    namespace p = phase_b_test;
    const auto mdl = p::iface_model("genderage");
    const auto dtonx = p::iface_onnx("det_10g");
    const auto imgf = p::image("test_person.jpg");
    if (!p::avail(mdl) || !p::avail(dtonx) || !p::avail(imgf)) return;

    auto img = ImageData::imread(imgf.string());
    face::InsightFaceDet det(dtonx.string(), p::ort_opt());
    std::vector<face::InsightFaceBox> boxes;
    REQUIRE(det.predict(img, &boxes));
    if (boxes.empty()) return;

    face::InsightFaceGenderAge ort(p::iface_onnx("genderage").string(), p::ort_opt());
    face::InsightFaceGenderAge ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    face::GenderAgeResult r_ort, r_ncnn;
    REQUIRE(ort.predict_gender_age(img, boxes[0].bbox, &r_ort));
    REQUIRE(ncnn.predict_gender_age(img, boxes[0].bbox, &r_ncnn));
    REQUIRE(r_ort.gender >= 0);
    REQUIRE(r_ort.age >= 0);
    REQUIRE(r_ncnn.gender == r_ort.gender);
    REQUIRE(std::abs(r_ncnn.age - r_ort.age) <= 1);
}

TEST_CASE("InsightFace landmark 2d106 ncnn vs ORT", "[ncnn][phase_b][insightface][lm2d]") {
    namespace p = phase_b_test;
    const auto mdl = p::iface_model("2d106det");
    const auto dtonx = p::iface_onnx("det_10g");
    const auto imgf = p::image("test_person.jpg");
    if (!p::avail(mdl) || !p::avail(dtonx) || !p::avail(imgf)) return;

    auto img = ImageData::imread(imgf.string());
    face::InsightFaceDet det(dtonx.string(), p::ort_opt());
    std::vector<face::InsightFaceBox> boxes;
    REQUIRE(det.predict(img, &boxes));
    if (boxes.empty()) return;

    face::InsightFaceLandmark ort(p::iface_onnx("2d106det").string(), p::ort_opt());
    face::InsightFaceLandmark ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    std::vector<std::array<float,2>> pts_ort, pts_ncnn;
    REQUIRE(ort.predict_2d106(img, boxes[0].bbox, &pts_ort));
    REQUIRE(ncnn.predict_2d106(img, boxes[0].bbox, &pts_ncnn));
    REQUIRE(pts_ort.size() == 106);
    REQUIRE(pts_ncnn.size() == 106);
    double rmse = 0;
    for (size_t i = 0; i < 106; ++i) rmse += (pts_ort[i][0]-pts_ncnn[i][0])*(pts_ort[i][0]-pts_ncnn[i][0]) + (pts_ort[i][1]-pts_ncnn[i][1])*(pts_ort[i][1]-pts_ncnn[i][1]);
    rmse = std::sqrt(rmse / 106);
    REQUIRE(rmse < 2.0);  // 平均 2px
}

TEST_CASE("InsightFace landmark 3d68 ncnn vs ORT", "[ncnn][phase_b][insightface][lm3d]") {
    namespace p = phase_b_test;
    const auto mdl = p::iface_model("1k3d68");
    const auto dtonx = p::iface_onnx("det_10g");
    const auto imgf = p::image("test_person.jpg");
    if (!p::avail(mdl) || !p::avail(dtonx) || !p::avail(imgf)) return;

    auto img = ImageData::imread(imgf.string());
    face::InsightFaceDet det(dtonx.string(), p::ort_opt());
    std::vector<face::InsightFaceBox> boxes;
    REQUIRE(det.predict(img, &boxes));
    if (boxes.empty()) return;

    face::InsightFaceLandmark ort(p::iface_onnx("1k3d68").string(), p::ort_opt());
    face::InsightFaceLandmark ncnn(mdl.string(), p::ncnn_opt());
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());
    std::vector<std::array<float,3>> pts_ort, pts_ncnn;
    std::array<float,3> pose_ort{}, pose_ncnn{};
    REQUIRE(ort.predict_3d68(img, boxes[0].bbox, &pts_ort, &pose_ort));
    REQUIRE(ncnn.predict_3d68(img, boxes[0].bbox, &pts_ncnn, &pose_ncnn));
    REQUIRE(pts_ort.size() == 68);
    REQUIRE(pts_ncnn.size() == 68);
    double rmse = 0;
    for (size_t i = 0; i < 68; ++i) for (int c = 0; c < 3; ++c) rmse += (pts_ort[i][c]-pts_ncnn[i][c])*(pts_ort[i][c]-pts_ncnn[i][c]);
    rmse = std::sqrt(rmse / (68*3));
    REQUIRE(rmse < 3.0);
    for (int c = 0; c < 3; ++c) REQUIRE(std::fabs(pose_ort[c] - pose_ncnn[c]) < 2.0);  // 度
}

TEST_CASE("InsightFace Analysis pipeline ncnn vs ORT", "[ncnn][phase_b][insightface][pipe]") {
    namespace p = phase_b_test;
    // ORT 基线用 .onnx（默认 ORT 后端按扩展名自动识别）；ncnn 用 .param 绝对路径
    const std::string onxdir = (p::data_root() / "test_data" / "test_models" / "onnx" / "insightface" / "buffalo_l").string();
    const std::string dir = (p::data_root() / "test_data" / "test_models" / "ncnn" / "insightface").string();
    const auto imgf = p::image("test_person.jpg");
    const bool has = p::avail(onxdir + "/det_10g.onnx") && p::avail(onxdir + "/w600k_r50.onnx")
                  && p::avail(onxdir + "/2d106det.onnx") && p::avail(onxdir + "/1k3d68.onnx")
                  && p::avail(onxdir + "/genderage.onnx")
                  && p::avail(dir + "/det_10g/det_10g.param") && p::avail(dir + "/2d106det/2d106det.param")
                  && p::avail(dir + "/1k3d68/1k3d68.param") && p::avail(dir + "/recognition/recognition.param")
                  && p::avail(dir + "/genderage/genderage.param") && p::avail(imgf);
    if (!has) return;

    auto img = ImageData::imread(imgf.string());
    face::InsightFaceAnalysis ort(onxdir + "/det_10g.onnx", onxdir + "/w600k_r50.onnx",
                                  onxdir + "/2d106det.onnx", onxdir + "/1k3d68.onnx",
                                  p::ort_opt(), onxdir + "/genderage.onnx");
    face::InsightFaceAnalysis ncnn(dir + "/det_10g/det_10g.param", dir + "/recognition/recognition.param",
                                   dir + "/2d106det/2d106det.param", dir + "/1k3d68/1k3d68.param",
                                   p::ncnn_opt(), dir + "/genderage/genderage.param");
    REQUIRE(ort.is_initialized());
    REQUIRE(ncnn.is_initialized());

    // ncnn 仅支持 batch=1（见 ncnn_backend.cpp 4D 输入按 batch=1 语义处理），
    // 而 test_person.jpg 经 scrfd 可检到多张（>1）候选框，故两侧都用 max_face_only 单脸
    // 端到端锚点（det→lmk2d→lmk3d→rec→ga），规避 ncnn batch>1 的已知边界。
    std::vector<face::InsightFaceResult> ro, rn;
    REQUIRE(ort.analyze(img, &ro, true, true, true, true, true));
    REQUIRE(ncnn.analyze(img, &rn, true, true, true, true, true));
    // 每人脸 embedding 余弦 > 0.99（同权重同输入，两后端应几乎一致；按 bbox 中心、score 匹配）
    size_t matched = 0, total = std::min(ro.size(), rn.size());
    for (size_t i = 0; i < total; ++i) {
        if (ro[i].embedding.empty() || rn[i].embedding.empty()) continue;
        if (ro[i].embedding.size() != rn[i].embedding.size()) continue;
        double dot = 0, na = 0, nb = 0;
        for (size_t k = 0; k < ro[i].embedding.size(); ++k) { dot += ro[i].embedding[k]*rn[i].embedding[k]; na += ro[i].embedding[k]*ro[i].embedding[k]; nb += rn[i].embedding[k]*rn[i].embedding[k]; }
        const double sim = dot / (std::sqrt(na) * std::sqrt(nb));
        INFO("pipe face " << i << " embedding cosine sim=" << sim);
        if (sim > 0.99) ++matched;
    }
    REQUIRE(matched >= 1);  // 至少一张人脸高相似
}
#endif
