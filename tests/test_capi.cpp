//
// capi2 设备帧/绘制相关回归测试（无需模型文件，CI 安全）
//
// 覆盖 Task 5 新增/改动：
//   - md_image_handle 统一持有 ImageData image（from_bgr24/from_nv12 均已填充）
//   - md_image_from_nv12 仍将 NV12 转为 CPU BGR（行为保持）
//   - md_image_plane_ptrs：NV12 才有效；CPU BGR 返回 UNSUPPORTED_TYPE 且 y/uv=NULL
//   - md_draw_result 对 CPU BGR 仍走 vis_*（就地绘制，写回底层内存）
//
// 注：真正的设备 NV12 帧由 md_image_from_device_nv12 + md_model_predict 产出，需模型文件（[model] 标签，CI 下载）。

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

TEST_CASE("capi2 nv12 from delegate: CPU BGR handle usable by md_draw_rect", "[capi]") {
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

    // CPU BGR handle 可直接被绘制接口使用（就地绘制，不走设备帧）
    MDColorRGBA color{255, 0, 0, 255};
    CHECK(md_draw_rect(img, 2, 2, 30, 20, color, 1.0f) == MD_OK);

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

    // pose / iseg 自省 names 也包含额外参数
    const char* pose_names = nullptr;
    REQUIRE(md_model_param_names(MD_MODEL_POSE, &pose_names) == MD_OK);
    CHECK(std::string(pose_names).find("keypoints_num") != std::string::npos);
    const char* iseg_names = nullptr;
    REQUIRE(md_model_param_names(MD_MODEL_INSTANCE_SEG, &iseg_names) == MD_OK);
    CHECK(std::string(iseg_names).find("mask_threshold") != std::string::npos);

    // OCR 整链路自省也包含 cls_thresh（经 get_classifier() 路由）
    const char* ocr_names = nullptr;
    REQUIRE(md_model_param_names(MD_MODEL_OCR, &ocr_names) == MD_OK);
    CHECK(std::string(ocr_names).find("cls_thresh") != std::string::npos);
    REQUIRE(md_model_param_type(MD_MODEL_OCR, "cls_thresh", &t) == MD_OK);
    CHECK(t == 'D');

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

TEST_CASE("capi2 option device id setter", "[capi]") {
    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);

    // 默认；set_device 与 set_device_id 独立调用均须不崩溃、可重复设置
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_device_id(opt, 0);
    md_option_set_device(opt, MD_DEV_GPU);
    md_option_set_device_id(opt, 1);
    md_option_set_device_id(opt, 2);
    // 负数收敛为 0（内部），仍可调用
    md_option_set_device_id(opt, -1);
    md_option_set_device_id(opt, 3);

    md_option_destroy(opt);
}

TEST_CASE("capi2 face anti-spoof enum + spoof getter guards", "[capi]") {
    // 新增 kind 在合法枚举范围内（不越界、不撞 MD_MODEL_COUNT）
    CHECK(static_cast<int>(MD_MODEL_FACE_AS_SECOND) < static_cast<int>(MD_MODEL_COUNT));

    // spoof getter 空指针 / 类型不符
    MDResultHandle res = nullptr;
    CHECK(md_result_spoof(nullptr, 0, nullptr) == MD_ERR_NULL_POINTER);
    CHECK(md_result_spoof(res, 0, nullptr) == MD_ERR_NULL_POINTER);

    // FACE_AS 用不存在的文件 create → 返回错误（不崩溃）
    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    MDModelHandle m = nullptr;
    CHECK(md_model_create(&m, MD_MODEL_FACE_AS, "no_such_fas.onnx", opt) != MD_OK);
    md_option_destroy(opt);
}

// 端到端人脸防伪：需模型文件（[model] 标签，CI 有模型时执行）
TEST_CASE("capi2 face anti-spoof inference (first)", "[model]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string model = data_dir + "/test_models/onnx/face/fas_first.onnx";
    const std::string imgf = data_dir + "/test_images/test_face_id3.jpg";
    if (!std::filesystem::exists(model) || !std::filesystem::exists(imgf)) return;

    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle as = nullptr;
    REQUIRE(md_model_create(&as, MD_MODEL_FACE_AS, model.c_str(), opt) == MD_OK);
    md_option_destroy(opt);

    MDImageHandle img = nullptr;
    REQUIRE(md_image_from_file(&img, imgf.c_str()) == MD_OK);
    MDResultHandle res = nullptr;
    REQUIRE(md_model_predict(as, img, &res) == MD_OK);
    size_t n = 0;
    REQUIRE(md_result_count(res, &n) == MD_OK);
    REQUIRE(n >= 1);
    int label = -1;
    REQUIRE(md_result_spoof(res, 0, &label) == MD_OK);
    CHECK(label >= 0);
    CHECK(label <= 2);

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(as);
}

TEST_CASE("capi2 image_from_device_nv12 wraps zero-copy two-plane, self-describes", "[capi]") {
    const int w = 16, h = 16;
    std::vector<unsigned char> y(w * h, 100), uv(w * h / 2, 100);
    MDImageHandle img = nullptr;
    REQUIRE(md_image_from_device_nv12(&img, y.data(), uv.data(), w, h, w, w, MD_DEV_CPU) == MD_OK);
    REQUIRE(img != nullptr);
    int ow = 0, oh = 0;
    REQUIRE(md_image_size(img, &ow, &oh) == MD_OK);
    CHECK(ow == w);
    CHECK(oh == h);
    MDDevice dev = MD_DEV_CPU; void* py = nullptr; void* puv = nullptr;
    REQUIRE(md_image_plane_ptrs(img, &dev, &py, &puv) == MD_OK);
    CHECK(dev == MD_DEV_CPU);
    CHECK(py == y.data());
    CHECK(puv == uv.data());
    md_image_destroy(img);
}

TEST_CASE("capi2 crop delegates to ImageData, preserves CPU/OOB/device semantics", "[capi]") {
    const int w = 16, h = 16;
    auto bgr = make_gray_bgr(w, h);
    MDImageHandle img = nullptr;
    REQUIRE(md_image_from_bgr24(&img, bgr.data(), w, h) == MD_OK);

    // CPU BGR 界内裁剪：结果尺寸正确
    MDImageHandle crop = nullptr;
    REQUIRE(md_image_crop(img, 2, 2, 6, 6, &crop) == MD_OK);
    REQUIRE(crop != nullptr);
    int cw = 0, ch = 0;
    REQUIRE(md_image_size(crop, &cw, &ch) == MD_OK);
    CHECK(cw == 6);
    CHECK(ch == 6);
    md_image_destroy(crop);

    // 界外裁剪：仍返回 MD_ERR_INVALID_ARGUMENT（行为保持）
    MDImageHandle oob = (MDImageHandle)0x1;
    CHECK(md_image_crop(img, 4, 4, 100, 100, &oob) == MD_ERR_INVALID_ARGUMENT);
    CHECK(oob == (MDImageHandle)0x1);  // out 未被写入
    md_image_destroy(img);

    // 设备 NV12 帧裁剪：MD_ERR_UNSUPPORTED_TYPE
    std::vector<unsigned char> y(w * h, 100), uv(w * h / 2, 100);
    MDImageHandle dev = nullptr;
    REQUIRE(md_image_from_device_nv12(&dev, y.data(), uv.data(), w, h, w, w, MD_DEV_CPU) == MD_OK);
    MDImageHandle dcrop = nullptr;
    CHECK(md_image_crop(dev, 0, 0, 4, 4, &dcrop) == MD_ERR_UNSUPPORTED_TYPE);
    md_image_destroy(dev);
}
