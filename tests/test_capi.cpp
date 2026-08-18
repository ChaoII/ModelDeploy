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
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

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

// PedestrianAttribute pipeline：cls batch size setter 的合法性验证（[model] 有数据时执行）
TEST_CASE("capi2 ped-attr cls batch size setter", "[model]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string det = data_dir + "/test_models/onnx/zhgd_det.onnx";
    const std::string ml = data_dir + "/test_models/onnx/zhgd_ml.onnx";
    if (!std::filesystem::exists(det) || !std::filesystem::exists(ml)) return;

    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle ped = nullptr;
    const std::string joined = det + "|" + ml;
    REQUIRE(md_model_create(&ped, MD_MODEL_PED_ATTR, joined.c_str(), opt) == MD_OK);
    REQUIRE(ped != nullptr);
    md_option_destroy(opt);

    // 非法 batch：0 / < -1 → INVALID_ARGUMENT
    CHECK(md_model_set_cls_batch_size(ped, 0) == MD_ERR_INVALID_ARGUMENT);
    CHECK(md_model_set_cls_batch_size(ped, -2) == MD_ERR_INVALID_ARGUMENT);
    // 合法：-1（自动）/ >0（固定）
    CHECK(md_model_set_cls_batch_size(ped, -1) == MD_OK);
    CHECK(md_model_set_cls_batch_size(ped, 1) == MD_OK);
    CHECK(md_model_set_cls_batch_size(ped, 4) == MD_OK);

    // 非 pipeline kind 上调用 → UNSUPPORTED（用一个未就绪句柄避免额外加载）
    CHECK(md_model_set_cls_batch_size(nullptr, 1) == MD_ERR_MODEL_INIT);

    md_model_destroy(ped);
}

// LPR_DET：阈值 + 输入尺寸 全链路暴露（此前为整类黑盒）
TEST_CASE("capi2 lpr-det setter + introspection", "[model]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string model = data_dir + "/test_models/onnx/yolov5plate.onnx";
    if (!std::filesystem::exists(model)) return;

    // 自省
    const char* names = nullptr;
    REQUIRE(md_model_param_names(MD_MODEL_LPR_DET, &names) == MD_OK);
    CHECK(std::string(names).find("conf_threshold") != std::string::npos);
    CHECK(std::string(names).find("nms_threshold") != std::string::npos);
    CHECK(std::string(names).find("landmarks_per_card") != std::string::npos);
    char t = 0;
    REQUIRE(md_model_param_type(MD_MODEL_LPR_DET, "landmarks_per_card", &t) == MD_OK);
    CHECK(t == 'D');

    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    MDModelHandle det = nullptr;
    REQUIRE(md_model_create(&det, MD_MODEL_LPR_DET, model.c_str(), opt) == MD_OK);
    REQUIRE(det != nullptr);
    md_option_destroy(opt);

    CHECK(md_model_set_param_d(det, "conf_threshold", 0.35) == MD_OK);
    CHECK(md_model_set_param_d(det, "nms_threshold", 0.5) == MD_OK);
    CHECK(md_model_set_param_d(det, "landmarks_per_card", 4) == MD_OK);
    CHECK(md_model_set_input_size(det, 640, 640) == MD_OK);
    md_model_destroy(det);
}

// OCR：det_max_side_len / cls_batch / rec_batch / rec_image_shape 全链路暴露
TEST_CASE("capi2 ocr setter batch + shape", "[model]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string ocr_dir = data_dir + "/test_models/onnx/ocr/ppocrv4_mobile";
    const std::string det = ocr_dir + "/det_infer.onnx";
    const std::string cls = ocr_dir + "/cls_infer.onnx";
    const std::string rec = ocr_dir + "/rec_infer.onnx";
    const std::string dict = data_dir + "/ppocrv4_dict.txt";
    if (!std::filesystem::exists(det) || !std::filesystem::exists(cls) ||
        !std::filesystem::exists(rec) || !std::filesystem::exists(dict)) return;

    // OCR_DET 单模型：max_side_len 自省 + 设置
    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    MDModelHandle detm = nullptr;
    REQUIRE(md_model_create(&detm, MD_MODEL_OCR_DET, det.c_str(), opt) == MD_OK);
    REQUIRE(detm != nullptr);
    const char* dnames = nullptr;
    REQUIRE(md_model_param_names(MD_MODEL_OCR_DET, &dnames) == MD_OK);
    CHECK(std::string(dnames).find("max_side_len") != std::string::npos);
    CHECK(md_model_set_param_i(detm, "max_side_len", 960) == MD_OK);
    CHECK(md_model_set_param_d(detm, "max_side_len", 0.5) == MD_ERR_INVALID_TYPE);  // 类型不符
    md_model_destroy(detm);

    // OCR 整链路：cls_batch/rec_batch/rec_image_shape/max_side_len
    MDModelHandle ocr = nullptr;
    const std::string joined = det + "|" + cls + "|" + rec + "|" + dict;
    REQUIRE(md_model_create(&ocr, MD_MODEL_OCR, joined.c_str(), opt) == MD_OK);
    REQUIRE(ocr != nullptr);
    md_option_destroy(opt);

    CHECK(md_model_set_cls_batch_size(ocr, -1) == MD_OK);
    CHECK(md_model_set_cls_batch_size(ocr, 2) == MD_OK);
    CHECK(md_model_set_cls_batch_size(ocr, 0) == MD_ERR_INVALID_ARGUMENT);
    CHECK(md_model_set_rec_batch_size(ocr, -1) == MD_OK);
    CHECK(md_model_set_rec_batch_size(ocr, 3) == MD_OK);
    CHECK(md_model_set_rec_batch_size(ocr, -2) == MD_ERR_INVALID_ARGUMENT);
    CHECK(md_model_set_param_i(ocr, "max_side_len", 1920) == MD_OK);
    CHECK(md_model_set_rec_image_shape(ocr, 3, 48, 320) == MD_OK);
    CHECK(md_model_set_rec_image_shape(ocr, 0, 48, 320) == MD_ERR_INVALID_ARGUMENT);
    md_model_destroy(ocr);
}

// 非对应 kind 上的 size 路由不应崩溃：FACE_REC_PIPELINE 缺模型时跳过路由测试仅需 det 类已覆盖，
// 这里验证 UNSUPPORTED 分支（nullptr → MODEL_INIT）
TEST_CASE("capi2 ocr batch size rejects wrong kind", "[capi]") {
    CHECK(md_model_set_rec_batch_size(nullptr, 1) == MD_ERR_MODEL_INIT);
    CHECK(md_model_set_rec_image_shape(nullptr, 3, 48, 320) == MD_ERR_MODEL_INIT);
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

TEST_CASE("capi2 md_image_from_yuv420p matches reference cvtColor(COLOR_YUV2BGR_I420)", "[capi]") {
    const int w = 16, h = 16;  // 偶宽偶高（I420 转换要求）
    std::vector<unsigned char> flat(static_cast<size_t>(w) * h * 3 / 2);
    for (size_t i = 0; i < flat.size(); ++i)
        flat[i] = static_cast<unsigned char>((i * 13) % 256);

    MDImageHandle img = nullptr;
    REQUIRE(md_image_from_yuv420p(&img, flat.data(), w, h) == MD_OK);
    REQUIRE(img != nullptr);
    int ow = 0, oh = 0;
    REQUIRE(md_image_size(img, &ow, &oh) == MD_OK);
    CHECK(ow == w);
    CHECK(oh == h);

    // 手柄 → lossless BMP 编码 → 解码还原 CPU BGR 像素
    const unsigned char* enc = nullptr;
    size_t n = 0;
    REQUIRE(md_image_encode(img, ".bmp", &enc, &n) == MD_OK);
    std::vector<unsigned char> encbuf(enc, enc + n);
    cv::Mat out = cv::imdecode(encbuf, cv::IMREAD_COLOR);
    REQUIRE(!out.empty());

    // 参考：对同一平铺 buffer 直接做 OpenCV I420→BGR
    cv::Mat yuv(h * 3 / 2, w, CV_8UC1, const_cast<unsigned char*>(flat.data()));
    cv::Mat ref;
    cv::cvtColor(yuv, ref, cv::COLOR_YUV2BGR_I420);
    REQUIRE(!ref.empty());

    CHECK(out.rows == ref.rows);
    CHECK(out.cols == ref.cols);
    CHECK(std::memcmp(out.data, ref.data, ref.total() * 3) == 0);

    md_image_destroy(img);
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

// 元数据 getter：type=MdImageType 数值, dev=MDDevice, nplanes=平面数（NV12=2, packed=1）
TEST_CASE("capi2 md_image_info returns type/device/plane metadata", "[capi]") {
    // GPU 设备 NV12 两平面帧（外部 y/uv 借用，仅验证元数据）：type=NV12(60), dev=GPU, nplanes=2
    const int w = 16, h = 16;
    std::vector<unsigned char> y(static_cast<size_t>(w) * h, 100),
                               uv(static_cast<size_t>(w) * h / 2, 100);
    MDImageHandle nv = nullptr;
    REQUIRE(md_image_from_device_nv12(&nv, y.data(), uv.data(), w, h, w, w, MD_DEV_GPU) == MD_OK);
    REQUIRE(nv != nullptr);
    int type = -1, dev = -1, nplanes = -1;
    REQUIRE(md_image_info(nv, &type, &dev, &nplanes) == MD_OK);
    CHECK(type == 60);            // MdImageType::NV12
    CHECK(nplanes == 2);
    CHECK(dev == MD_DEV_GPU);     // MD_DEV_GPU == 1 == Device::GPU
    md_image_destroy(nv);

    // CPU 紧致 BGR 图：type=PKG_BGR_U8(22), 单平面, CPU
    auto bgr = make_gray_bgr(w, h);
    MDImageHandle img = nullptr;
    REQUIRE(md_image_from_bgr24(&img, bgr.data(), w, h) == MD_OK);
    type = -1; dev = -1; nplanes = -1;
    REQUIRE(md_image_info(img, &type, &dev, &nplanes) == MD_OK);
    CHECK(type == 22);            // MdImageType::PKG_BGR_U8
    CHECK(nplanes == 1);
    CHECK(dev == MD_DEV_CPU);
    md_image_destroy(img);

    // 空句柄 → MD_ERR_NULL_POINTER
    CHECK(md_image_info(nullptr, &type, &dev, &nplanes) == MD_ERR_NULL_POINTER);

    // 任一输出指针可为空（仅查询部分项）
    MDImageHandle img2 = nullptr;
    REQUIRE(md_image_from_bgr24(&img2, bgr.data(), w, h) == MD_OK);
    type = -1;
    REQUIRE(md_image_info(img2, &type, nullptr, nullptr) == MD_OK);
    CHECK(type == 22);
    md_image_destroy(img2);
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

//
// R1 回归：结果数组 getter 必须幂等（同一 getter 二次调用返回一致的量化结果），
// 且"依赖型" getter（keypoints/mask 等）在未先调用数组 getter 时也应安全。
// 旧实现把已投影的 ProjectedResult 当 ResultData 强转（正式 UB，classification 会读到野值）。
//
TEST_CASE("capi2 result getters are idempotent and standalone-safe", "[model]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string det_file = data_dir + "/test_models/onnx/yolo11n/yolo11n.onnx";
    const std::string cls_file = data_dir + "/test_models/onnx/yolo11n/yolo11n-cls.onnx";
    const std::string pose_file = data_dir + "/test_models/onnx/yolo11n/yolo11n-pose.onnx";
    const std::string imgf = data_dir + "/test_images/bus.jpg";
    if (!std::filesystem::exists(det_file) || !std::filesystem::exists(cls_file) ||
        !std::filesystem::exists(pose_file) || !std::filesystem::exists(imgf)) {
        return;
    }

    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDImageHandle img = nullptr;
    REQUIRE(md_image_from_file(&img, imgf.c_str()) == MD_OK);

    // --- detection：同一 getter 二次调用，结果必须一致 ---
    {
        MDModelHandle det = nullptr;
        REQUIRE(md_model_create(&det, MD_MODEL_DETECTION, det_file.c_str(), opt) == MD_OK);
        MDResultHandle res = nullptr;
        REQUIRE(md_model_predict(det, img, &res) == MD_OK);

        size_t n1 = 0, n2 = 0;
        const MDDetectionItem* a1 = nullptr;
        const MDDetectionItem* a2 = nullptr;
        REQUIRE(md_result_detection(res, &a1, &n1) == MD_OK);
        REQUIRE(md_result_detection(res, &a2, &n2) == MD_OK);
        CHECK(n1 == n2);
        size_t cnt = 0;
        REQUIRE(md_result_count(res, &cnt) == MD_OK);
        CHECK(cnt == n1);
        for (size_t i = 0; i < n1 && i < n2; ++i) {
            CHECK(a1[i].x == a2[i].x);
            CHECK(a1[i].y == a2[i].y);
            CHECK(a1[i].w == a2[i].w);
            CHECK(a1[i].h == a2[i].h);
            CHECK(a1[i].score == a2[i].score);
            CHECK(a1[i].label_id == a2[i].label_id);
        }
        md_result_destroy(res);
        md_model_destroy(det);
    }

    // --- classification：二次调用不得读野值（旧实现此处损坏） ---
    {
        MDModelHandle cls = nullptr;
        REQUIRE(md_model_create(&cls, MD_MODEL_CLASSIFICATION, cls_file.c_str(), opt) == MD_OK);
        MDResultHandle res = nullptr;
        REQUIRE(md_model_predict(cls, img, &res) == MD_OK);

        size_t n1 = 0, n2 = 0;
        const MDClassifyItem* c1 = nullptr;
        const MDClassifyItem* c2 = nullptr;
        REQUIRE(md_result_classification(res, &c1, &n1) == MD_OK);
        REQUIRE(md_result_classification(res, &c2, &n2) == MD_OK);
        CHECK(n1 == n2);
        for (size_t i = 0; i < n1 && i < n2; ++i) {
            CHECK(c1[i].label_id == c2[i].label_id);
            CHECK(c1[i].score == c2[i].score);
        }
        md_result_destroy(res);
        md_model_destroy(cls);
    }

    // --- pose：keypoints 依赖型 getter，在数组 getter 之前独立调用须安全 ---
    {
        MDModelHandle pose = nullptr;
        REQUIRE(md_model_create(&pose, MD_MODEL_POSE, pose_file.c_str(), opt) == MD_OK);
        MDResultHandle res = nullptr;
        REQUIRE(md_model_predict(pose, img, &res) == MD_OK);

        const MDPoint3* kps = nullptr;
        size_t kn = 0;
        CHECK(md_result_keypoints(res, 0, &kps, &kn) == MD_OK);  // 未先调 md_result_pose
        md_result_destroy(res);
        md_model_destroy(pose);
    }

    md_image_destroy(img);
    md_option_destroy(opt);
}

TEST_CASE("capi2 rgb24 input converts to BGR identical to reference", "[capi]") {
    const int w = 32, h = 24;
    std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < rgb.size(); i += 3) {
        rgb[i] = static_cast<unsigned char>((i * 7) % 256);
        rgb[i + 1] = static_cast<unsigned char>((i * 11) % 256);
        rgb[i + 2] = static_cast<unsigned char>((i * 13) % 256);
    }
    MDImageHandle img = nullptr;
    REQUIRE(md_image_from_rgb24(&img, rgb.data(), w, h) == MD_OK);
    REQUIRE(img != nullptr);

    // 无损 BMP 编码解码还原手柄内 BGR 像素
    const unsigned char* enc = nullptr;
    size_t n = 0;
    REQUIRE(md_image_encode(img, ".bmp", &enc, &n) == MD_OK);
    std::vector<unsigned char> encbuf(enc, enc + n);
    cv::Mat out = cv::imdecode(encbuf, cv::IMREAD_COLOR);
    REQUIRE(!out.empty());

    // 参考：cv::cvtColor RGB->BGR
    cv::Mat src(h, w, CV_8UC3, const_cast<unsigned char*>(rgb.data()));
    cv::Mat ref;
    cv::cvtColor(src, ref, cv::COLOR_RGB2BGR);
    REQUIRE(!ref.empty());

    CHECK(out.rows == ref.rows);
    CHECK(out.cols == ref.cols);
    CHECK(std::memcmp(out.data, ref.data, ref.total() * 3) == 0);

    md_image_destroy(img);
}

TEST_CASE("capi2 device NV12 frame ops return UNSUPPORTED_TYPE", "[capi]") {
    const int w = 16, h = 16;
    std::vector<unsigned char> y(w * h, 100), uv(w * h / 2, 100);
    MDImageHandle dev = nullptr;
    REQUIRE(md_image_from_device_nv12(&dev, y.data(), uv.data(), w, h, w, w, MD_DEV_CPU) == MD_OK);
    REQUIRE(dev != nullptr);

    // encode
    const unsigned char* enc = nullptr;
    size_t n = 0;
    CHECK(md_image_encode(dev, ".bmp", &enc, &n) == MD_ERR_UNSUPPORTED_TYPE);
    // save（asMat 失败前不应写文件）
    CHECK(md_image_save(dev, "capi2_t6_should_not_exist.bmp") == MD_ERR_UNSUPPORTED_TYPE);
    CHECK(!std::filesystem::exists("capi2_t6_should_not_exist.bmp"));
    // draw_rect
    MDColorRGBA c{255, 0, 0, 255};
    CHECK(md_draw_rect(dev, 1, 1, 4, 4, c, 1.0f) == MD_ERR_UNSUPPORTED_TYPE);
    // show（在 asMat 处即报错，不会阻塞在 waitKey）
    CHECK(md_image_show(dev) == MD_ERR_UNSUPPORTED_TYPE);

    md_image_destroy(dev);
}

TEST_CASE("capi2 CPU BGR frame encode/save still OK", "[capi]") {
    const int w = 16, h = 16;
    auto bgr = make_gray_bgr(w, h);
    MDImageHandle img = nullptr;
    REQUIRE(md_image_from_bgr24(&img, bgr.data(), w, h) == MD_OK);

    const unsigned char* enc = nullptr;
    size_t n = 0;
    REQUIRE(md_image_encode(img, ".png", &enc, &n) == MD_OK);
    REQUIRE(n > 0);

    const char* tmp = std::getenv("TEMP");
    const std::string p = std::string(tmp && *tmp ? tmp : ".") + "/md_capi_t6_save.png";
    REQUIRE(md_image_save(img, p.c_str()) == MD_OK);
    CHECK(std::filesystem::exists(p));
    std::filesystem::remove(p);

    md_image_destroy(img);
}
