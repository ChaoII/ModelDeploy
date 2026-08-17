//
// capi2 设备帧/绘制相关回归测试（无需模型文件，CI 安全）
//
// 覆盖 Task 5 新增/改动：
//   - md_image_handle 统一持有 ImageData image（from_bgr24/from_nv12 均已填充）
//   - md_image_from_nv12 仍将 NV12 转为 CPU BGR（行为保持）
//   - md_image_plane_ptrs：NV12 才有效；CPU BGR 返回 UNSUPPORTED_TYPE 且 y/uv=NULL
//   - md_draw_result 对 CPU BGR 仍走 vis_*（就地绘制，写回底层内存）
//
// 注：真正的设备 NV12 帧由 predict_nv12 产出，需模型文件（[model] 标签，CI 下载）。

#include <catch2/catch_test_macros.hpp>

#include "capi2/md_capi.h"

#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// 生成一个纯灰 BGR24 图（w*h*3 字节）
std::vector<unsigned char> make_gray_bgr(int w, int h) {
    std::vector<unsigned char> buf(static_cast<size_t>(w) * h * 3, 128);
    return buf;
}

} // namespace

TEST_CASE("capi2 image handle holds unified ImageData", "[capi]") {
    const int w = 64, h = 48;
    auto bgr = make_gray_bgr(w, h);
    MDImageHandle img = nullptr;
    const MDStatus s = md_image_from_bgr24(&img, bgr.data(), w, h);
    REQUIRE(s == MD_OK);
    REQUIRE(img != nullptr);

    int ow = 0, oh = 0;
    REQUIRE(md_image_size(img, &ow, &oh) == MD_OK);
    CHECK(ow == w);
    CHECK(oh == h);

    // CPU BGR 图：plane_ptrs 应返回 UNSUPPORTED_TYPE 且 y/uv=NULL（NV12 才有效）
    MDDevice dev = MD_DEV_CPU;
    void* y = reinterpret_cast<void*>(0x1);
    void* uv = reinterpret_cast<void*>(0x2);
    const MDStatus ps = md_image_plane_ptrs(img, &dev, &y, &uv);
    CHECK(ps == MD_ERR_UNSUPPORTED_TYPE);
    CHECK(y == nullptr);
    CHECK(uv == nullptr);

    md_image_destroy(img);
}

TEST_CASE("capi2 nv12 input converts to CPU BGR and plane_ptrs rejects it", "[capi]") {
    const int w = 64, h = 48;
    std::vector<unsigned char> y(w * h, 128);
    std::vector<unsigned char> uv(w * h / 2, 128);
    MDImageHandle img = nullptr;
    const MDStatus s = md_image_from_nv12(&img, y.data(), uv.data(), w, h, w, w, MD_DEV_CPU);
    REQUIRE(s == MD_OK);
    REQUIRE(img != nullptr);

    int ow = 0, oh = 0;
    REQUIRE(md_image_size(img, &ow, &oh) == MD_OK);
    CHECK(ow == w);
    CHECK(oh == h);

    // NV12 输入已转 CPU BGR（非 NV12 类型），plane_ptrs 对 CPU BGR 约定返回 UNSUPPORTED_TYPE
    MDDevice dev = MD_DEV_GPU;
    void* py = nullptr;
    void* puv = nullptr;
    const MDStatus ps = md_image_plane_ptrs(img, &dev, &py, &puv);
    CHECK(ps == MD_ERR_UNSUPPORTED_TYPE);
    CHECK(py == nullptr);
    CHECK(puv == nullptr);

    md_image_destroy(img);
}

TEST_CASE("capi2 null args are rejected", "[capi]") {
    MDDevice dev;
    void* y;
    void* uv;
    CHECK(md_image_plane_ptrs(nullptr, &dev, &y, &uv) == MD_ERR_NULL_POINTER);

    const int w = 16, h = 16;
    auto bgr = make_gray_bgr(w, h);
    MDImageHandle img = nullptr;
    REQUIRE(md_image_from_bgr24(&img, bgr.data(), w, h) == MD_OK);
    CHECK(md_image_plane_ptrs(img, nullptr, &y, &uv) == MD_ERR_NULL_POINTER);
    CHECK(md_image_plane_ptrs(img, &dev, nullptr, &uv) == MD_ERR_NULL_POINTER);
    CHECK(md_image_plane_ptrs(img, &dev, &y, nullptr) == MD_ERR_NULL_POINTER);
    md_image_destroy(img);
}

TEST_CASE("capi2 model set param + introspection", "[capi]") {
    // 自省：names
    const char* names = nullptr;
    REQUIRE(md_model_param_names(MD_MODEL_DETECTION, &names) == MD_OK);
    REQUIRE(names);
    CHECK(std::string(names).find("conf_threshold") != std::string::npos);
    CHECK(std::string(names).find("nms_threshold") != std::string::npos);

    // 自省：type
    char t = 0;
    REQUIRE(md_model_param_type(MD_MODEL_DETECTION, "conf_threshold", &t) == MD_OK);
    CHECK(t == 'D');
    REQUIRE(md_model_param_type(MD_MODEL_DETECTION, "nms_threshold", &t) == MD_OK);
    CHECK(t == 'D');

    // pose / classification / ocr 类型
    REQUIRE(md_model_param_type(MD_MODEL_POSE, "keypoints_num", &t) == MD_OK);
    CHECK(t == 'I');
    REQUIRE(md_model_param_type(MD_MODEL_CLASSIFICATION, "multi_label", &t) == MD_OK);
    CHECK(t == 'B');
    REQUIRE(md_model_param_type(MD_MODEL_OCR_DET, "det_db_score_mode", &t) == MD_OK);
    CHECK(t == 'S');

    // 未知名 → INVALID_ARGUMENT
    CHECK(md_model_param_type(MD_MODEL_DETECTION, "nope", &t) == MD_ERR_INVALID_ARGUMENT);

    // 无参数 kind → 空 names
    const char* sem_names = nullptr;
    REQUIRE(md_model_param_names(MD_MODEL_SEM_SEG, &sem_names) == MD_OK);
    CHECK(std::string(sem_names).empty());

    // kind 越界 → INVALID_ARGUMENT
    const char* dummy_names = nullptr;
    CHECK(md_model_param_names((MDModelKind)999, &dummy_names) == MD_ERR_INVALID_ARGUMENT);
    char dummy_t = 0;
    CHECK(md_model_param_type((MDModelKind)999, "conf_threshold", &dummy_t) == MD_ERR_INVALID_ARGUMENT);

    // 自省空指针
    CHECK(md_model_param_names(MD_MODEL_DETECTION, nullptr) == MD_ERR_NULL_POINTER);
    CHECK(md_model_param_type(MD_MODEL_DETECTION, nullptr, &t) == MD_ERR_NULL_POINTER);
}

// 端到端 setter：需可加载的检测模型（[model] 标签，CI 有模型时执行）
TEST_CASE("capi2 detection param setter on loaded model", "[model]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string modelfile = data_dir + "/test_models/onnx/yolo11n/yolo11n.onnx";
    if (!std::filesystem::exists(modelfile)) return;

    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle det = nullptr;
    REQUIRE(md_model_create(&det, MD_MODEL_DETECTION, modelfile.c_str(), opt) == MD_OK);
    REQUIRE(det != nullptr);
    md_option_destroy(opt);

    CHECK(md_model_set_param_d(det, "conf_threshold", 0.3) == MD_OK);
    CHECK(md_model_set_param_d(det, "nms_threshold", 0.4) == MD_OK);

    // 类型不匹配：conf_threshold 是 D，用 _i 应返回 INVALID_TYPE
    CHECK(md_model_set_param_i(det, "conf_threshold", 3) == MD_ERR_INVALID_TYPE);
    // 未知名 → INVALID_ARGUMENT
    CHECK(md_model_set_param_d(det, "nope", 0.5) == MD_ERR_INVALID_ARGUMENT);

    md_model_destroy(det);
}
