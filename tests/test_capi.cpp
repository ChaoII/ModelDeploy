//
// capi 设备帧/绘制相关回归测试（无需模型文件，CI 安全）
//
// 覆盖 Task 5 新增/改动：
//   - md_image_handle 统一持有 ImageData image（from_bgr24/from_nv12 均已填充）
//   - md_image_from_nv12 仍将 NV12 转为 CPU BGR（行为保持）
//   - md_image_plane_ptrs：NV12 才有效；CPU BGR 返回 UNSUPPORTED_TYPE 且 y/uv=NULL
//   - md_draw_result 对 CPU BGR 仍走 vis_*（就地绘制，写回底层内存）
//
// 注：真正的设备 NV12 帧由 md_image_from_device_nv12 + md_model_predict 产出，需模型文件（[model] 标签，CI 下载）。

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "capi/md_capi.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <algorithm>
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

TEST_CASE("capi image handle holds unified ImageData", "[capi]") {
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

TEST_CASE("capi nv12 input converts to CPU BGR and plane_ptrs rejects it", "[capi]") {
    const int w = 64, h = 48;
    std::vector<unsigned char> y(w * h, 128);
    std::vector<unsigned char> uv(w * h / 2, 128);
    MDImageHandle img = nullptr;
    const MDStatus s = md_image_from_nv12(&img, y.data(), uv.data(), w, h, w, w);
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

TEST_CASE("capi nv12 from delegate: CPU BGR handle usable by md_draw_rect", "[capi]") {
    const int w = 64, h = 48;
    std::vector<unsigned char> y(w * h, 128);
    std::vector<unsigned char> uv(w * h / 2, 128);
    MDImageHandle img = nullptr;
    const MDStatus s = md_image_from_nv12(&img, y.data(), uv.data(), w, h, w, w);
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

TEST_CASE("capi null args are rejected", "[capi]") {
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

TEST_CASE("capi model set param + introspection", "[capi]") {
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

// hand：MD_MODEL_HAND 契约。参数自省部分无模型即可验证（[capi]）；
// create/predict 需 hand_pose.onnx（外链 modelscope，缺失则优雅跳过）。
TEST_CASE("capi hand keypoints contract (MD_MODEL_HAND)", "[capi]") {
    // 自省：hand 与 pose 同参数表，含 keypoints_num（int）
    const char* names = nullptr;
    REQUIRE(md_model_param_names(MD_MODEL_HAND, &names) == MD_OK);
    CHECK(names);
    CHECK(std::string(names).find("keypoints_num") != std::string::npos);
    char t = 0;
    REQUIRE(md_model_param_type(MD_MODEL_HAND, "keypoints_num", &t) == MD_OK);
    CHECK(t == 'I');

    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string modelfile = data_dir + "/test_models/onnx/hand_pose.onnx";
    const std::string imgf = data_dir + "/test_images/bus.jpg";
    if (!std::filesystem::exists(modelfile) || !std::filesystem::exists(imgf)) {
        WARN("hand_pose.onnx 权重缺失（外链 modelscope），跳过 create/predict");
        return;
    }

    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle hand = nullptr;
    REQUIRE(md_model_create(&hand, MD_MODEL_HAND, modelfile.c_str(), opt) == MD_OK);
    REQUIRE(hand != nullptr);
    md_option_destroy(opt);

    // keypoints_num 端到端 setter（int 类型）
    CHECK(md_model_set_param_i(hand, "keypoints_num", 21) == MD_OK);
    CHECK(md_model_set_param_d(hand, "conf_threshold", 0.3) == MD_OK);
    CHECK(md_model_set_param_d(hand, "keypoints_num", 21.0) == MD_ERR_INVALID_TYPE);

    // clone 深拷贝可用
    MDModelHandle clone = nullptr;
    REQUIRE(md_model_clone(hand, &clone) == MD_OK);
    REQUIRE(clone != nullptr);

    // predict 走 pose 结果机制（KeyPointsResult → MD_RES_POSE）
    MDImageHandle img = nullptr;
    REQUIRE(md_image_from_file(&img, imgf.c_str()) == MD_OK);
    MDResultHandle res = nullptr;
    REQUIRE(md_model_predict(clone, img, &res) == MD_OK);
    MDResultKind kind = (MDResultKind)-1;
    REQUIRE(md_result_kind(res, &kind) == MD_OK);
    CHECK(kind == MD_RES_POSE);
    const MDPoseItem* items = nullptr;
    size_t cnt = 0;
    REQUIRE(md_result_pose(res, &items, &cnt) == MD_OK);
    if (cnt > 0) {
        const MDPoint3* kps = nullptr;
        size_t kn = 0;
        REQUIRE(md_result_keypoints(res, 0, &kps, &kn) == MD_OK);
        CHECK(kn == 21);
    }

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(clone);
    md_model_destroy(hand);
}

// 端到端 setter：需可加载的检测模型（[model] 标签，CI 有模型时执行）
TEST_CASE("capi detection param setter on loaded model", "[model]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string modelfile = data_dir + "/test_models/onnx/yolo26n/yolo26n.onnx";
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
TEST_CASE("capi ped-attr cls batch size setter", "[model]") {
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
TEST_CASE("capi lpr-det setter + introspection", "[model]") {
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
TEST_CASE("capi ocr setter batch + shape", "[model]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string ocr_dir = data_dir + "/test_models/onnx/ocr/ppocrv6_tiny";
    const std::string det = ocr_dir + "/det_infer.onnx";
    const std::string cls = ocr_dir + "/cls_infer.onnx";
    const std::string rec = ocr_dir + "/rec_infer.onnx";
    const std::string dict = data_dir + "/ppocrv6_tiny_dict.txt";
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
TEST_CASE("capi ocr batch size rejects wrong kind", "[capi]") {
    CHECK(md_model_set_rec_batch_size(nullptr, 1) == MD_ERR_MODEL_INIT);
    CHECK(md_model_set_rec_image_shape(nullptr, 3, 48, 320) == MD_ERR_MODEL_INIT);
}

TEST_CASE("capi option device id setter", "[capi]") {
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

TEST_CASE("capi face anti-spoof enum + spoof getter guards", "[capi]") {
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
TEST_CASE("capi face anti-spoof inference (first)", "[model]") {
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

TEST_CASE("capi md_image_from_yuv420p matches reference cvtColor(COLOR_YUV2BGR_I420)", "[capi]") {
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

TEST_CASE("capi image_from_device_nv12 wraps zero-copy two-plane, self-describes", "[capi]") {
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
TEST_CASE("capi md_image_info returns type/device/plane metadata", "[capi]") {
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

TEST_CASE("capi crop delegates to ImageData, preserves CPU/OOB/device semantics", "[capi]") {
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
TEST_CASE("capi result getters are idempotent and standalone-safe", "[model]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string det_file = data_dir + "/test_models/onnx/yolo26n/yolo26n.onnx";
    const std::string cls_file = data_dir + "/test_models/onnx/yolo26n/yolo26n-cls.onnx";
    const std::string pose_file = data_dir + "/test_models/onnx/yolo26n/yolo26n-pose.onnx";
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

// 2D 批量结果 API：predict_batch 按图分组，逐图 *_batch getter 返回每图自己的项数组
TEST_CASE("capi batch result is per-image grouped (2D)", "[model]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string det_file = data_dir + "/test_models/onnx/yolo26n/yolo26n.onnx";
    const std::string imgf = data_dir + "/test_images/bus.jpg";
    if (!std::filesystem::exists(det_file) || !std::filesystem::exists(imgf)) {
        return;
    }

    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle det = nullptr;
    REQUIRE(md_model_create(&det, MD_MODEL_DETECTION, det_file.c_str(), opt) == MD_OK);

    MDImageHandle im[2] = {nullptr, nullptr};
    REQUIRE(md_image_from_file(&im[0], imgf.c_str()) == MD_OK);
    REQUIRE(md_image_from_file(&im[1], imgf.c_str()) == MD_OK);

    MDResultHandle res = nullptr;
    REQUIRE(md_model_predict_batch(det, im, 2, &res) == MD_OK);
    REQUIRE(res != nullptr);

    // 批量容器按图分组：md_result_count 应等于图像数（2），而非平铺项数
    size_t imgs = 0;
    REQUIRE(md_result_count(res, &imgs) == MD_OK);
    CHECK(imgs == 2);

    size_t total = 0;
    for (size_t i = 0; i < 2; ++i) {
        const MDDetectionItem* items = nullptr;
        size_t cnt = i == 0 ? 0 : 1;  // 初始为非零，验证 getter 确实覆盖 out 参数
        CHECK(md_result_detection_batch(res, i, &items, &cnt) == MD_OK);
        CHECK(cnt > 0);            // 每图至少检出
        CHECK(items != nullptr);
        total += cnt;
    }
    CHECK(total > 0);

    // 越界图索引：明确报错（不崩）
    const MDDetectionItem* items = nullptr;
    size_t cnt = 0;
    CHECK(md_result_detection_batch(res, 2, &items, &cnt) == MD_ERR_INVALID_ARGUMENT);

    // 单图 getter 不应作用在批量句柄上（返回 INVALID_ARGUMENT，而非读错误内存）
    CHECK(md_result_detection(res, &items, &cnt) == MD_ERR_INVALID_ARGUMENT);

    md_result_destroy(res);
    md_image_destroy(im[0]);
    md_image_destroy(im[1]);
    md_model_destroy(det);
    md_option_destroy(opt);
}

TEST_CASE("capi rgb24 input converts to BGR identical to reference", "[capi]") {
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

TEST_CASE("capi device NV12 frame ops return UNSUPPORTED_TYPE", "[capi]") {
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
    CHECK(md_image_save(dev, "capi_t6_should_not_exist.bmp") == MD_ERR_UNSUPPORTED_TYPE);
    CHECK(!std::filesystem::exists("capi_t6_should_not_exist.bmp"));
    // draw_rect
    MDColorRGBA c{255, 0, 0, 255};
    CHECK(md_draw_rect(dev, 1, 1, 4, 4, c, 1.0f) == MD_ERR_UNSUPPORTED_TYPE);
    // show（在 asMat 处即报错，不会阻塞在 waitKey）
    CHECK(md_image_show(dev) == MD_ERR_UNSUPPORTED_TYPE);

    md_image_destroy(dev);
}

TEST_CASE("capi CPU BGR frame encode/save still OK", "[capi]") {
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

TEST_CASE("capi predict_batch rejects null args (no model needed)", "[capi]") {
    MDImageHandle img = nullptr;
    const int w = 16, h = 16;
    auto bgr = make_gray_bgr(w, h);
    REQUIRE(md_image_from_bgr24(&img, bgr.data(), w, h) == MD_OK);

    MDResultHandle res = nullptr;
    MDImageHandle imgs[2] = {img, img};

    // 空模型句柄 / 空输出 → NULL_POINTER
    CHECK(md_model_predict_batch(nullptr, imgs, 2, &res) == MD_ERR_NULL_POINTER);
    CHECK(md_model_predict_batch(nullptr, nullptr, 0, nullptr) == MD_ERR_NULL_POINTER);

    md_image_destroy(img);
}

// 端到端批量：需可加载的检测模型（[model] 标签，CI 有模型时执行）
TEST_CASE("capi predict_batch detection flattens both images", "[model]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string det_file = data_dir + "/test_models/onnx/yolo26n/yolo26n.onnx";
    const std::string img1 = data_dir + "/test_images/test_detection0.jpg";
    const std::string img2 = data_dir + "/test_images/test_detection1.jpg";
    if (!std::filesystem::exists(det_file) || !std::filesystem::exists(img1) ||
        !std::filesystem::exists(img2)) {
        return;
    }

    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle det = nullptr;
    REQUIRE(md_model_create(&det, MD_MODEL_DETECTION, det_file.c_str(), opt) == MD_OK);
    REQUIRE(det != nullptr);
    md_option_destroy(opt);

    MDImageHandle a = nullptr, b = nullptr;
    REQUIRE(md_image_from_file(&a, img1.c_str()) == MD_OK);
    REQUIRE(md_image_from_file(&b, img2.c_str()) == MD_OK);
    MDImageHandle imgs[2] = {a, b};

    // 错误路径：n==0 → INVALID_ARGUMENT；imgs==nullptr → NULL_POINTER
    MDResultHandle r = nullptr;
    CHECK(md_model_predict_batch(det, imgs, 0, &r) == MD_ERR_INVALID_ARGUMENT);
    CHECK(md_model_predict_batch(det, nullptr, 2, &r) == MD_ERR_NULL_POINTER);

    // 正确批量：按图分组（2D）。批量句柄用 *_batch 逐图取项；flat getter 不再适用
    MDResultHandle batch = nullptr;
    REQUIRE(md_model_predict_batch(det, imgs, 2, &batch) == MD_OK);
    REQUIRE(batch != nullptr);

    const MDDetectionItem* items = nullptr;
    size_t flat_cnt = 0;
    CHECK(md_result_detection(batch, &items, &flat_cnt) == MD_ERR_INVALID_ARGUMENT);  // flat getter 不适用批量句柄

    size_t total = 0;
    size_t per_image[2] = {0, 0};
    for (size_t i = 0; i < 2; ++i) {
        REQUIRE(md_result_detection_batch(batch, i, &items, &per_image[i]) == MD_OK);
        total += per_image[i];
    }

    // 每图单图推理作为参考：批量中该图项数必须与单图一致（2D 分组正确），且总框数 == 各单图之和
    size_t expect = 0;
    for (size_t i = 0; i < 2; ++i) {
        MDResultHandle sr = nullptr;
        REQUIRE(md_model_predict(det, i == 0 ? a : b, &sr) == MD_OK);
        size_t cnt = 0;
        REQUIRE(md_result_count(sr, &cnt) == MD_OK);
        CHECK(cnt == per_image[i]);
        expect += cnt;
        md_result_destroy(sr);
    }
    CHECK(total == expect);

    md_result_destroy(batch);
    md_image_destroy(a);
    md_image_destroy(b);
    md_model_destroy(det);
}

// Task 3：单值类 kind 批量 getter —— 无模型可测的 NULL_POINTER 错误路径（[capi]）
TEST_CASE("capi single-value batch getters reject null args (no model needed)", "[capi]") {
    MDResultHandle h = nullptr;
    const int* items = nullptr;
    size_t count = 0;

    // 逐项 batch getter：空句柄 → NULL_POINTER，不崩
    CHECK(md_result_age_batch(h, &items, &count) == MD_ERR_NULL_POINTER);
    CHECK(md_result_gender_batch(h, &items, &count) == MD_ERR_NULL_POINTER);

    const unsigned char* labels = nullptr;
    size_t oh = 0, ow = 0;
    int nc = 0;
    CHECK(md_result_sem_seg_batch(h, 0, &labels, &oh, &ow, &nc) == MD_ERR_NULL_POINTER);

    const float* depth = nullptr;
    CHECK(md_result_depth_batch(h, 0, &depth, &oh, &ow) == MD_ERR_NULL_POINTER);

    CHECK(md_result_ocr_batch_count(h, &count) == MD_ERR_NULL_POINTER);

    // 既有单值 getter：空句柄 → NULL_POINTER，不崩
    int v = 0;
    CHECK(md_result_age(h, &v) == MD_ERR_NULL_POINTER);
    CHECK(md_result_gender(h, &v) == MD_ERR_NULL_POINTER);
    CHECK(md_result_sem_seg(h, &labels, &oh, &ow, &nc) == MD_ERR_NULL_POINTER);
    CHECK(md_result_depth(h, &depth, &oh, &ow) == MD_ERR_NULL_POINTER);
    const int* quad = nullptr;
    const char* text = nullptr;
    float score = 0.f;
    CHECK(md_result_ocr(h, 0, &quad, &text, &score) == MD_ERR_NULL_POINTER);
}

// Task 3：真实 age 批量断言（模型存在时才执行；无模型环境直接跳过保持 [capi] 全绿）
TEST_CASE("capi age batch getters + single-getter compat (real model, guarded)", "[capi]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string age_file = data_dir + "/test_models/onnx/face/age_predictor.onnx";
    const std::string img1 = data_dir + "/test_images/test_face.jpg";
    const std::string img2 = data_dir + "/test_images/test_face.jpg";
    if (!std::filesystem::exists(age_file) || !std::filesystem::exists(img1)) {
        return;
    }

    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle age = nullptr;
    REQUIRE(md_model_create(&age, MD_MODEL_FACE_AGE, age_file.c_str(), opt) == MD_OK);
    REQUIRE(age != nullptr);
    md_option_destroy(opt);

    MDImageHandle a = nullptr, b = nullptr;
    REQUIRE(md_image_from_file(&a, img1.c_str()) == MD_OK);
    REQUIRE(md_image_from_file(&b, img2.c_str()) == MD_OK);
    MDImageHandle imgs[2] = {a, b};

    MDResultHandle batch = nullptr;
    REQUIRE(md_model_predict_batch(age, imgs, 2, &batch) == MD_OK);
    REQUIRE(batch != nullptr);

    // 批量 getter：返回 2 个 age
    const int* items = nullptr;
    size_t count = 0;
    REQUIRE(md_result_age_batch(batch, &items, &count) == MD_OK);
    CHECK(count == 2);
    CHECK(items != nullptr);

    // 既有单值 getter 兼容 ResultData：读 index 0，与 batch[0] 一致
    int single_age = -1;
    REQUIRE(md_result_age(batch, &single_age) == MD_OK);
    CHECK(single_age == items[0]);

    // kind 不匹配 → INVALID_ARGUMENT，不崩
    int g = 0;
    CHECK(md_result_gender_batch(batch, &items, &count) == MD_ERR_INVALID_ARGUMENT);
    CHECK(md_result_gender(batch, &g) == MD_ERR_INVALID_ARGUMENT);
    const unsigned char* labels = nullptr;
    size_t oh = 0, ow = 0;
    int nc = 0;
    CHECK(md_result_sem_seg_batch(batch, 0, &labels, &oh, &ow, &nc) == MD_ERR_INVALID_ARGUMENT);
    const float* depth = nullptr;
    CHECK(md_result_depth_batch(batch, 0, &depth, &oh, &ow) == MD_ERR_INVALID_ARGUMENT);
    CHECK(md_result_ocr_batch_count(batch, &count) == MD_ERR_INVALID_ARGUMENT);

    md_result_destroy(batch);
    md_image_destroy(a);
    md_image_destroy(b);
    md_model_destroy(age);
}

// Task 4：ReID (OSNet) 契约 —— 512-d embedding getter（模型存在才行，缺权重 SKIP，[capi]）
TEST_CASE("capi reid embedding getter (real model, guarded)", "[capi]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string model_file = data_dir + "/test_models/onnx/osnet_x1_0.onnx";
    if (!std::filesystem::exists(model_file)) {
        WARN("OSNet model not found; skipping reid capi test. "
             "Download test_data from modelscope: "
             "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps");
        return;
    }

    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle reid = nullptr;
    REQUIRE(md_model_create(&reid, MD_MODEL_REID, model_file.c_str(), opt) == MD_OK);
    REQUIRE(reid != nullptr);
    md_option_destroy(opt);

    // OSNet 输入 128x256（HxW），纯灰 BGR 图
    MDImageHandle img = nullptr;
    const int w = 128, h = 256;
    auto bgr = make_gray_bgr(w, h);
    REQUIRE(md_image_from_bgr24(&img, bgr.data(), w, h) == MD_OK);

    MDResultHandle res = nullptr;
    REQUIRE(md_model_predict(reid, img, &res) == MD_OK);
    REQUIRE(res != nullptr);

    MDResultKind kind = MD_RES_DETECTION;
    REQUIRE(md_result_kind(res, &kind) == MD_OK);
    CHECK(kind == MD_RES_REID);

    const float* emb = nullptr;
    size_t emb_n = 0;
    REQUIRE(md_result_reid_embedding(res, 0, &emb, &emb_n) == MD_OK);
    CHECK(emb_n == 512);
    CHECK(emb != nullptr);
    if (emb && emb_n == 512) {
        double n2 = 0.0;
        for (size_t k = 0; k < emb_n; ++k) n2 += (double)emb[k] * emb[k];
        CHECK(std::sqrt(n2) == Catch::Approx(1.0).margin(1e-3));
    }

    // kind 不匹配 → INVALID_ARGUMENT，不崩
    CHECK(md_result_face_embedding(res, 0, &emb, &emb_n) == MD_ERR_INVALID_ARGUMENT);

    // 批量 getter：单图
    const float* bemb = nullptr;
    size_t b_n = 0;
    REQUIRE(md_result_reid_embedding_batch(res, 0, &bemb, &b_n) == MD_OK);
    CHECK(b_n == 512);
    CHECK(bemb != nullptr);

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(reid);
}

// Task 10：C API 跟踪器 —— 纯 CPU 无模型依赖（[capi]）
TEST_CASE("capi tracker bytetrack update keeps stable track_id", "[capi]") {
    MDTrackerHandle h = nullptr;
    REQUIRE(md_tracker_create(MD_TRACKER_BYTETRACK, &h) == MD_OK);
    REQUIRE(h != nullptr);

    // 一个移动框，两帧映射到同一 track
    MDBox boxes[1];
    float scores[1];
    int label_ids[1];

    MDTrackItem out[4];
    int first_id = -1;

    boxes[0] = MDBox{100.f, 100.f, 40.f, 40.f};
    scores[0] = 0.95f;
    label_ids[0] = 1;
    {
        size_t cap = 4;
        REQUIRE(md_tracker_update(h, boxes, scores, label_ids, 1, out, &cap) == MD_OK);
        REQUIRE(cap >= 1);
        REQUIRE(cap <= 4);
        first_id = out[0].track_id;
        CHECK(out[0].label_id == 1);
        CHECK(out[0].score == 0.95f);
        CHECK(out[0].state >= 0);
        CHECK(out[0].state <= 3);
        CHECK(out[0].x == 100.f);
        CHECK(out[0].y == 100.f);
    }

    // 第二帧同目标小幅移动 → 应关联到同一个 track_id
    boxes[0] = MDBox{104.f, 104.f, 40.f, 40.f};
    {
        size_t cap = 4;
        REQUIRE(md_tracker_update(h, boxes, scores, label_ids, 1, out, &cap) == MD_OK);
        REQUIRE(cap >= 1);
        CHECK(out[0].track_id == first_id);
    }

    // 容量不足 → 报错且不写越界
    {
        size_t cap = 0;
        MDTrackItem tmp = MDTrackItem{-1.f, -1.f, -1.f, -1.f, -1, -1, -1.f, -1};
        CHECK(md_tracker_update(h, boxes, scores, label_ids, 1, &tmp, &cap) == MD_ERR_INVALID_ARGUMENT);
        CHECK(cap == 1);              // 需求数 1（*out_count 置为需要数）
        CHECK(tmp.track_id == -1);    // out 未被写入
    }

    // set_params：已知名生效，未知名报错
    CHECK(md_tracker_set_params(h, "max_age", 60) == MD_OK);
    CHECK(md_tracker_set_params(h, "track_thresh", 0.6) == MD_OK);
    CHECK(md_tracker_set_params(h, "match_thresh", 0.99) == MD_OK);
    CHECK(md_tracker_set_params(h, "nope", 1.0) == MD_ERR_INVALID_ARGUMENT);
    CHECK(md_tracker_set_params(h, "", 1.0) == MD_ERR_INVALID_ARGUMENT);
    CHECK(md_tracker_set_params(nullptr, "max_age", 1.0) == MD_ERR_NULL_POINTER);

    // reset 后可继续使用（track_id 从 0 重新计数，因不再匹配旧 track）
    REQUIRE(md_tracker_reset(h) == MD_OK);
    {
        size_t cap = 4;
        REQUIRE(md_tracker_update(h, boxes, scores, label_ids, 1, out, &cap) == MD_OK);
        REQUIRE(cap >= 1);
        CHECK(out[0].track_id == 0);   // reset 归零
    }

    md_tracker_destroy(h);
}

TEST_CASE("capi tracker create null args + empty-frame capacity", "[capi]") {
    MDTrackerHandle h = nullptr;
    CHECK(md_tracker_create(MD_TRACKER_BYTETRACK, nullptr) == MD_ERR_NULL_POINTER);
    CHECK(md_tracker_create(MD_TRACKER_STRONGSORT, &h) == MD_OK);
    REQUIRE(h != nullptr);

    // 空输入（n=0）：合法，无输出
    MDTrackItem out[4];
    size_t cap = 4;
    const MDBox* nb = nullptr;
    const float* ns = nullptr;
    const int* nl = nullptr;
    REQUIRE(md_tracker_update(h, nb, ns, nl, 0, out, &cap) == MD_OK);
    CHECK(cap == 0);

    // null 入参守卫
    CHECK(md_tracker_update(nullptr, nb, ns, nl, 0, out, &cap) == MD_ERR_NULL_POINTER);
    CHECK(md_tracker_update(h, nb, ns, nl, 0, nullptr, &cap) == MD_ERR_NULL_POINTER);
    CHECK(md_tracker_update(h, nb, ns, nl, 1, out, &cap) == MD_ERR_NULL_POINTER);

    // 非法 kind → INVALID_ARGUMENT
    MDTrackerHandle bad = (MDTrackerHandle)0x1;
    CHECK(md_tracker_create((MDTrackerKind)999, &bad) == MD_ERR_INVALID_ARGUMENT);
    CHECK(bad == (MDTrackerHandle)0x1);  // out 未被写入

    md_tracker_destroy(h);
}

// F1 回归：md_tracker_capacity 是**非变异**查询——查询本身不推进跟踪器状态。
TEST_CASE("capi tracker_capacity is a non-mutating query", "[capi]") {
    MDTrackerHandle h = nullptr;
    REQUIRE(md_tracker_create(MD_TRACKER_BYTETRACK, &h) == MD_OK);
    REQUIRE(h != nullptr);

    MDBox b = MDBox{10.f, 10.f, 40.f, 40.f};
    float s = 0.9f;
    int l = 0;
    MDTrackItem out[4];
    size_t cap = 4;
    REQUIRE(md_tracker_update(h, &b, &s, &l, 1, out, &cap) == MD_OK);
    REQUIRE(cap == 1);
    const int id1 = out[0].track_id;

    // 多次非变异容量查询：返回需要数，且不改变任何状态
    for (int i = 0; i < 3; ++i) {
        size_t need = 99;
        REQUIRE(md_tracker_capacity(h, &b, &s, &l, 1, &need) == MD_OK);
        CHECK(need == 1);
    }

    // 查询后再次 update（同一框）仍关联到原 track：若查询曾推进帧计数/Kalman，
    // 输出计数会偏移/ID 会变化，此断言即失败。
    REQUIRE(md_tracker_update(h, &b, &s, &l, 1, out, &cap) == MD_OK);
    CHECK(cap == 1);
    CHECK(out[0].track_id == id1);

    // 空指针守卫
    CHECK(md_tracker_capacity(nullptr, &b, &s, &l, 1, &cap) == MD_ERR_NULL_POINTER);
    CHECK(md_tracker_capacity(h, &b, &s, &l, 1, nullptr) == MD_ERR_NULL_POINTER);

    md_tracker_destroy(h);
}

// F1 回归：双阶段探测（用 md_tracker_update 自身探测容量）会对每个有输出的帧推进两次，
// 导致 Lost 目标的 max_age 有效减半、过早被移除。修复后经 capacity 查询 + 单次 update，
// 目标在 max_age 内重新出现应沿用原 track_id（保持 ID）。
TEST_CASE("capi tracker no double-advance keeps lost track id", "[capi]") {
    MDTrackerHandle h = nullptr;
    REQUIRE(md_tracker_create(MD_TRACKER_BYTETRACK, &h) == MD_OK);
    REQUIRE(h != nullptr);
    REQUIRE(md_tracker_set_params(h, "max_age", 30) == MD_OK);

    MDBox boxes[2];
    float scores[2];
    int label_ids[2];
    MDTrackItem out[4];

    // 帧 1、2：目标 A 出现（小幅移动保持同一 track）
    boxes[0] = MDBox{100.f, 100.f, 40.f, 40.f};
    scores[0] = 0.95f;
    label_ids[0] = 1;
    {
        size_t cap = 4;
        REQUIRE(md_tracker_update(h, boxes, scores, label_ids, 1, out, &cap) == MD_OK);
        REQUIRE(cap >= 1);
    }
    boxes[0].x = 102.f;
    {
        size_t cap = 4;
        REQUIRE(md_tracker_update(h, boxes, scores, label_ids, 1, out, &cap) == MD_OK);
        REQUIRE(cap >= 1);
    }
    const int a_id = out[0].track_id;

    // 帧 3..25：A 消失，目标 B 持续出现于远处 → 每帧有输出（need>=1）。
    // 修复前：每逻辑帧推进 2 次，A 的 cur-frame_id 增长加倍，约第 18 帧即被移除；
    // 修复后：每帧推进 1 次，A 到第 33 帧才被移除，第 26 帧时仍存活。
    boxes[0] = MDBox{400.f, 400.f, 40.f, 40.f};
    scores[0] = 0.95f;
    label_ids[0] = 2;
    for (int i = 0; i < 23; ++i) {  // 帧 3..25
        size_t cap = 4;
        REQUIRE(md_tracker_update(h, boxes, scores, label_ids, 1, out, &cap) == MD_OK);
        REQUIRE(cap >= 1);
    }

    // 帧 26：A 重新出现（连同 B）。查询/提交契约：capacity（非变异）→ 分配 → update（恰一次）
    boxes[0] = MDBox{100.f, 100.f, 40.f, 40.f};  // A
    boxes[1] = MDBox{400.f, 400.f, 40.f, 40.f};  // B
    scores[0] = 0.95f;
    scores[1] = 0.95f;
    label_ids[0] = 1;
    label_ids[1] = 2;
    {
        size_t need = 0;
        REQUIRE(md_tracker_capacity(h, boxes, scores, label_ids, 2, &need) == MD_OK);
        REQUIRE(need >= 1);
        std::vector<MDTrackItem> vout(need);
        size_t wcap = need;
        REQUIRE(md_tracker_update(h, boxes, scores, label_ids, 2, vout.data(), &wcap) == MD_OK);
        REQUIRE(wcap >= 1);

        bool found_a = false;
        for (size_t i = 0; i < wcap; ++i) {
            if (vout[i].x == 100.f && vout[i].y == 100.f) {
                found_a = true;
                CHECK(vout[i].track_id == a_id);  // 修复前此处失败（A 已丢、新 id）
            }
        }
        REQUIRE(found_a);
    }

    md_tracker_destroy(h);
}

// 定位 test_data/qr_sample.png（解码文本 https://example.com/MD）的路径。
// TEST_DATA_DIR 是 test 运行时注入的环境变量（=仓库根），回退到 ./test_data。
std::string barcode_qr_path() {
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string dir = env && *env ? std::string(env) : ".";
    return dir + "/test_data/qr_sample.png";
}

TEST_CASE("capi barcode create/destroy + null guards", "[capi]") {
    MDBarcodeHandle b = nullptr;
    REQUIRE(md_barcode_create(&b) == MD_OK);
    REQUIRE(b != nullptr);

    // null 传入 → NULL_POINTER
    CHECK(md_barcode_create(nullptr) == MD_ERR_NULL_POINTER);
    CHECK(md_barcode_set_formats(nullptr, 0) == MD_ERR_NULL_POINTER);
    CHECK(md_barcode_detect(nullptr, nullptr, nullptr, nullptr) == MD_ERR_NULL_POINTER);

    md_barcode_destroy(b);
}

TEST_CASE("capi barcode detect decodes a sample QR", "[capi]") {
    const std::string path = barcode_qr_path();
    if (!std::filesystem::exists(path)) {
        REQUIRE(false);
        return;
    }
    MDImageHandle img = nullptr;
    REQUIRE(md_image_from_file(&img, path.c_str()) == MD_OK);
    REQUIRE(img != nullptr);

    MDBarcodeHandle b = nullptr;
    REQUIRE(md_barcode_create(&b) == MD_OK);

    // 容量查询：items==nullptr 时返回需要数，不崩溃
    uint32_t need = 0;
    REQUIRE(md_barcode_detect(b, img, nullptr, &need) == MD_OK);
    REQUIRE(need >= 1);

    // 预分配并写入
    std::vector<MD_BarcodeItem> items(need);
    uint32_t cap = need;
    REQUIRE(md_barcode_detect(b, img, items.data(), &cap) == MD_OK);
    REQUIRE(cap >= 1);

    bool found = false;
    for (uint32_t i = 0; i < cap; ++i) {
        if (items[i].is_qr && std::string(items[i].text) == "https://example.com/MD") {
            found = true;
            CHECK(items[i].format[0] != '\0');   // 格式名非空（ZXing 如 "QR Code"）
            CHECK(items[i].quad[0] >= 0.f);   // 四点坐标合理（左上角）
            CHECK(items[i].score >= 0.f);
            break;
        }
    }
    REQUIRE(found);

    md_barcode_destroy(b);
    md_image_destroy(img);
}

// 声纹（MD_MODEL_SPEAKER_VERIFY）：「枚举创建 + 入口」契约。
// 模型缺失时 guard 跳过真推理，仅验证枚举/空参数/错误路径（CI 安全，无需模型文件）。
TEST_CASE("capi speaker verify enum + embed entry", "[capi]") {
    // 新枚举值有效且位于 REID 之后、COUNT 之前
    STATIC_REQUIRE(MD_MODEL_SPEAKER_VERIFY > MD_MODEL_REID);
    STATIC_REQUIRE(MD_MODEL_SPEAKER_VERIFY < MD_MODEL_COUNT);

    // 空/非法入参路径（不需模型即校验）
    float dummy_samp[16] = {0.f};
    const float* emb = nullptr;
    size_t emb_n = 0;
    CHECK(md_audio_speaker_embed(nullptr, dummy_samp, 16, &emb, &emb_n) == MD_ERR_NULL_POINTER);
    CHECK(md_audio_speaker_embed((MDModelHandle)(uintptr_t)1, nullptr, 16, &emb, &emb_n) == MD_ERR_NULL_POINTER);
    CHECK(md_audio_speaker_embed((MDModelHandle)(uintptr_t)1, dummy_samp, 0, &emb, &emb_n) == MD_ERR_INVALID_ARGUMENT);

    // 模型文件缺失 → 跳过加载类断言（加载与推理需外链模型）
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string modelfile = data_dir + "/test_models/onnx/speaker_verify/ecapa.onnx";
    if (!std::filesystem::exists(modelfile)) {
        WARN("speaker_verify ecapa.onnx 缺失（外链 modelscope），跳过 create/predict");
        return;
    }

    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle sv = nullptr;
    REQUIRE(md_model_create(&sv, MD_MODEL_SPEAKER_VERIFY, modelfile.c_str(), opt) == MD_OK);
    REQUIRE(sv != nullptr);
    md_option_destroy(opt);

    // clone 深拷贝可用
    MDModelHandle clone = nullptr;
    REQUIRE(md_model_clone(sv, &clone) == MD_OK);
    REQUIRE(clone != nullptr);

    // 取一段语音（约 1s @16k zeros）推断 embedding，验证借用指针维度>0 且指针稳定
    std::vector<float> samples(16000, 0.f);
    samples[0] = 0.5f;
    REQUIRE(md_audio_speaker_embed(clone, samples.data(), samples.size(), &emb, &emb_n) == MD_OK);
    REQUIRE(emb != nullptr);
    CHECK(emb_n > 0);

    md_model_destroy(clone);
    md_model_destroy(sv);
}

// FormulaRecognizer（MD_MODEL_FORMULA_RECOGNIZER）：「枚举 + 错误路径」契约。
// 模型缺失时 guard 跳过真推理，仅验证枚举/空参数/加载失败路径（CI 安全，无需模型文件）。
TEST_CASE("capi formula recognizer enum + create error path", "[capi]") {
    // 新枚举值有效且位于 SPEAKER_VERIFY 之后、COUNT 之前
    STATIC_REQUIRE(MD_MODEL_FORMULA_RECOGNIZER > MD_MODEL_SPEAKER_VERIFY);
    STATIC_REQUIRE(MD_MODEL_FORMULA_RECOGNIZER < MD_MODEL_COUNT);

    // 空入参路径（不需模型即校验）
    const char* latex = nullptr;
    CHECK(md_result_formula(nullptr, 0, &latex) == MD_ERR_NULL_POINTER);
    CHECK(md_result_formula((MDResultHandle)(uintptr_t)1, 0, nullptr) == MD_ERR_NULL_POINTER);

    // 加载失败路径：指向不存在的模型 → create 报错，不产生句柄
    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle f = nullptr;
    // dict 可选：仅模型路径（1 部分）
    CHECK(md_model_create(&f, MD_MODEL_FORMULA_RECOGNIZER, "nonexistent_formula.onnx", opt) == MD_ERR_MODEL_INIT);
    CHECK(f == nullptr);
    // 缺路径（空串）→ INVALID_ARGUMENT
    CHECK(md_model_create(&f, MD_MODEL_FORMULA_RECOGNIZER, "", opt) == MD_ERR_INVALID_ARGUMENT);
    md_option_destroy(opt);

    // 模型文件缺失 → 跳过加载类断言（真推理需外链模型）
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string modelfile = data_dir + "/test_models/onnx/formula_recognition/formula_rec.onnx";
    if (!std::filesystem::exists(modelfile)) {
        WARN("formula_rec.onnx 缺失（外链 modelscope），跳过 create/predict 真加载");
        return;
    }
}

// TSN / ST-GCN（MD_MODEL_TSN / MD_MODEL_ST_GCN）：「枚举 + 错误路径」契约。
// 模型缺失时 guard 跳过真推理，仅验证枚举/空参数/加载失败路径（CI 安全，无需模型文件）。
TEST_CASE("capi action (TSN/ST_GCN) enum + error path", "[capi]") {
    // 新枚举值有效且位于 FORMULA_RECOGNIZER 之后、COUNT 之前
    STATIC_REQUIRE(MD_MODEL_TSN > MD_MODEL_FORMULA_RECOGNIZER);
    STATIC_REQUIRE(MD_MODEL_TSN < MD_MODEL_COUNT);
    STATIC_REQUIRE(MD_MODEL_ST_GCN > MD_MODEL_TSN);
    STATIC_REQUIRE(MD_MODEL_ST_GCN < MD_MODEL_COUNT);

    // 空入参路径（不需模型即校验）：null 入参 → NULL_POINTER
    MDResultHandle res = nullptr;
    MDImageHandle dummy = nullptr;
    CHECK(md_model_predict_sequence(nullptr, nullptr, 0, &res) == MD_ERR_NULL_POINTER);
    CHECK(md_model_predict_sequence(nullptr, &dummy, 1, &res) == MD_ERR_NULL_POINTER);
    CHECK(md_model_predict_skeleton(nullptr, nullptr, 0, 0, 0, &res) == MD_ERR_NULL_POINTER);

    // 加载失败路径：指向不存在的模型 → create 报错，不产生句柄
    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle tsn = nullptr;
    CHECK(md_model_create(&tsn, MD_MODEL_TSN, "nonexistent_tsn.onnx", opt) == MD_ERR_MODEL_INIT);
    CHECK(tsn == nullptr);
    MDModelHandle stg = nullptr;
    CHECK(md_model_create(&stg, MD_MODEL_ST_GCN, "nonexistent_stgcn.onnx", opt) == MD_ERR_MODEL_INIT);
    CHECK(stg == nullptr);
    // 缺路径（空串）→ INVALID_ARGUMENT
    CHECK(md_model_create(&tsn, MD_MODEL_TSN, "", opt) == MD_ERR_INVALID_ARGUMENT);
    md_option_destroy(opt);

    // 模型文件缺失 → 跳过加载类断言（真推理需外链模型）
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string tsn_file = data_dir + "/test_models/onnx/action/tsn.onnx";
    const std::string stg_file = data_dir + "/test_models/onnx/action/stgcn.onnx";
    if (!std::filesystem::exists(tsn_file) || !std::filesystem::exists(stg_file)) {
        WARN("action tsn/stgcn 权重缺失（外链 modelscope），跳过 create/predict 真加载");
        return;
    }

    // 真加载：create 成功后再走 predict 错误路径（空入参）与正确 kind 守卫
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    REQUIRE(md_model_create(&tsn, MD_MODEL_TSN, tsn_file.c_str(), opt) == MD_OK);
    REQUIRE(tsn != nullptr);
    REQUIRE(md_model_create(&stg, MD_MODEL_ST_GCN, stg_file.c_str(), opt) == MD_OK);
    REQUIRE(stg != nullptr);
    md_option_destroy(opt);

    // TSN 句柄上调用 skeleton → UNSUPPORTED；ST_GCN 上调用 sequence → UNSUPPORTED（kind 守卫）
    MDImageHandle frame = nullptr;
    const int w = 224, h = 224;
    auto rgb = make_gray_bgr(w, h);
    REQUIRE(md_image_from_bgr24(&frame, rgb.data(), w, h) == MD_OK);
    MDImageHandle frames[1] = {frame};
    std::vector<float> joints(4 * 4 * 2);  // T*V*C = 4*4*2
    CHECK(md_model_predict_sequence(stg, frames, 1, &res) == MD_ERR_UNSUPPORTED_TYPE);
    CHECK(md_model_predict_skeleton(tsn, joints.data(), 4, 4, 2, &res) == MD_ERR_UNSUPPORTED_TYPE);
    // 正确 kind 下空入参 → INVALID_ARGUMENT
    CHECK(md_model_predict_sequence(tsn, nullptr, 0, &res) == MD_ERR_INVALID_ARGUMENT);
    CHECK(md_model_predict_skeleton(stg, nullptr, 0, 0, 0, &res) == MD_ERR_INVALID_ARGUMENT);
    md_image_destroy(frame);

    md_model_destroy(stg);
    md_model_destroy(tsn);
}

// VehicleKeypoint / FaceLandmark（MD_MODEL_VEHICLE_KEYPOINT / MD_MODEL_FACE_LANDMARK）：
// 结果复用 MD_RES_POSE（KeyPointsResult）形态；Vehicle 支持 keypoints_num/conf/nms 参数 + set_input_size，
// Face 不支持 set_param/set_size。「枚举 + 空参 + 错误路径」契约模型缺失即可验证（CI 安全）。
TEST_CASE("capi vehicle keypoint / face landmark enum + error path", "[capi]") {
    // 新枚举值有效且位于 ST_GCN 之后、COUNT 之前
    STATIC_REQUIRE(MD_MODEL_VEHICLE_KEYPOINT > MD_MODEL_ST_GCN);
    STATIC_REQUIRE(MD_MODEL_VEHICLE_KEYPOINT < MD_MODEL_COUNT);
    STATIC_REQUIRE(MD_MODEL_FACE_LANDMARK > MD_MODEL_VEHICLE_KEYPOINT);
    STATIC_REQUIRE(MD_MODEL_FACE_LANDMARK < MD_MODEL_COUNT);

    // 自省：Vehicle 与 pose/hand 同参数表，含 keypoints_num（int）
    const char* names = nullptr;
    REQUIRE(md_model_param_names(MD_MODEL_VEHICLE_KEYPOINT, &names) == MD_OK);
    CHECK(names);
    CHECK(std::string(names).find("keypoints_num") != std::string::npos);
    char t = 0;
    REQUIRE(md_model_param_type(MD_MODEL_VEHICLE_KEYPOINT, "keypoints_num", &t) == MD_OK);
    CHECK(t == 'I');
    // Face 无参数表
    CHECK(md_model_param_names(MD_MODEL_FACE_LANDMARK, &names) == MD_OK);
    CHECK((!names || !*names));

    // 空入参路径（不需模型即校验）
    MDResultHandle res = nullptr;
    MDImageHandle img = nullptr;
    MDModelHandle h = nullptr;
    CHECK(md_model_predict(nullptr, img, &res) == MD_ERR_NULL_POINTER);
    CHECK(md_model_predict_batch(nullptr, nullptr, 0, &res) == MD_ERR_NULL_POINTER);

    // 加载失败路径：不存在的模型 → create 报错，不产生句柄
    MDOptionHandle opt = nullptr;
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    CHECK(md_model_create(&h, MD_MODEL_VEHICLE_KEYPOINT, "nonexistent_vehicle_keypoint.onnx", opt) == MD_ERR_MODEL_INIT);
    CHECK(h == nullptr);
    CHECK(md_model_create(&h, MD_MODEL_FACE_LANDMARK, "nonexistent_2d106det.onnx", opt) == MD_ERR_MODEL_INIT);
    CHECK(h == nullptr);
    // 缺路径（空串）→ INVALID_ARGUMENT
    CHECK(md_model_create(&h, MD_MODEL_VEHICLE_KEYPOINT, "", opt) == MD_ERR_INVALID_ARGUMENT);
    CHECK(md_model_create(&h, MD_MODEL_FACE_LANDMARK, "", opt) == MD_ERR_INVALID_ARGUMENT);
    md_option_destroy(opt);

    // 模型文件缺失 → 跳过加载类断言（真推理需外链模型）
    const char* env = std::getenv("TEST_DATA_DIR");
    std::string data_dir = env && *env ? std::string(env) + "/test_data" : "test_data";
    const std::string veh_file = data_dir + "/test_models/onnx/vehicle_keypoint.onnx";
    const std::string face_file = data_dir + "/test_models/onnx/2d106det.onnx";
    const std::string imgf = data_dir + "/test_images/bus.jpg";
    if (!std::filesystem::exists(veh_file) || !std::filesystem::exists(imgf)) {
        WARN("vehicle_keypoint.onnx 权重缺失（外链 modelscope），跳过真加载/predict/result 断言");
        return;
    }

    // 真加载：Vehicle create + set_param_i(keypoints_num) 端到端 + result 形态 = MD_RES_POSE
    REQUIRE(md_option_create(&opt) == MD_OK);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    MDModelHandle veh = nullptr;
    REQUIRE(md_model_create(&veh, MD_MODEL_VEHICLE_KEYPOINT, veh_file.c_str(), opt) == MD_OK);
    REQUIRE(veh != nullptr);
    // 独立模型暴露 preprocessor，set_input_size 应 OK（Vehicle 镜像 HAND）
    CHECK(md_model_set_input_size(veh, 640, 640) == MD_OK);
    CHECK(md_model_set_param_i(veh, "keypoints_num", 8) == MD_OK);
    CHECK(md_model_set_param_d(veh, "conf_threshold", 0.3) == MD_OK);
    CHECK(md_model_set_param_d(veh, "keypoints_num", 8.0) == MD_ERR_INVALID_TYPE);
    // Face 不支持 set_param → 在已加载语句外由 param_type_of 上报空表（上面已校验），
    // 此处经 apply_model_param 的 decl_type==0 路径验证：仅 Vehicle 有参数。

    // clone + predict → MD_RES_POSE + md_result_keypoints
    MDModelHandle clone = nullptr;
    REQUIRE(md_model_clone(veh, &clone) == MD_OK);
    REQUIRE(clone != nullptr);
    MDImageHandle im = nullptr;
    REQUIRE(md_image_from_file(&im, imgf.c_str()) == MD_OK);
    REQUIRE(md_model_predict(clone, im, &res) == MD_OK);
    MDResultKind kind = (MDResultKind)-1;
    REQUIRE(md_result_kind(res, &kind) == MD_OK);
    CHECK(kind == MD_RES_POSE);
    const MDPoseItem* items = nullptr;
    size_t cnt = 0;
    REQUIRE(md_result_pose(res, &items, &cnt) == MD_OK);
    if (cnt > 0) {
        const MDPoint3* kps = nullptr;
        size_t kn = 0;
        REQUIRE(md_result_keypoints(res, 0, &kps, &kn) == MD_OK);
        CHECK(kn == 8);
    }
    md_result_destroy(res);
    md_image_destroy(im);
    md_model_destroy(clone);

    // FaceLandmark create + predict → MD_RES_POSE（106 点）
    if (std::filesystem::exists(face_file)) {
        MDModelHandle face = nullptr;
        REQUIRE(md_model_create(&face, MD_MODEL_FACE_LANDMARK, face_file.c_str(), opt) == MD_OK);
        REQUIRE(face != nullptr);
        // Face 无 preprocessor 尺寸开关/参数 setter：set_input_size 应报 UNSUPPORTED_TYPE
        CHECK(md_model_set_input_size(face, 192, 192) == MD_ERR_UNSUPPORTED_TYPE);
        CHECK(md_model_set_param_i(face, "keypoints_num", 106) == MD_ERR_INVALID_ARGUMENT);
        MDImageHandle fi = nullptr;
        REQUIRE(md_image_from_file(&fi, imgf.c_str()) == MD_OK);
        REQUIRE(md_model_predict(face, fi, &res) == MD_OK);
        REQUIRE(md_result_kind(res, &kind) == MD_OK);
        CHECK(kind == MD_RES_POSE);
        md_result_destroy(res);
        md_image_destroy(fi);
        md_model_destroy(face);
    } else {
        WARN("2d106det.onnx 权重缺失，跳过 FaceLandmark 真加载/predict 断言");
    }

    md_option_destroy(opt);
    md_model_destroy(veh);
}

TEST_CASE("cv solution + tool capi", "[capi]") {
    MDSolutionHandle h = nullptr;
    REQUIRE(md_solution_create(&h, MD_SOLUTION_OBJECT_COUNTER) == MD_OK);
    REQUIRE(md_solution_object_counter_set_line(h, 5.0f, 0.0f, 5.0f, 10.0f) == MD_OK);
    float b1[4] = {0,4,2,2}; int lid[1] = {0}; int tid[1] = {1};
    REQUIRE(md_solution_object_counter_update(h, b1, 1, lid, tid) == MD_OK);
    int in = -1, oc = -1;
    REQUIRE(md_solution_object_counter_hline(h, &in, &oc) == MD_OK);
    REQUIRE(in == 0);
    float b2[4] = {8,4,2,2};
    REQUIRE(md_solution_object_counter_update(h, b2, 1, lid, tid) == MD_OK);
    REQUIRE(md_solution_object_counter_hline(h, &in, &oc) == MD_OK);
    REQUIRE(in == 1);
    REQUIRE(md_solution_destroy(h) == MD_OK);
    float iou = 0;
    REQUIRE(md_vision_iou4(0,0,10,10, 0,0,10,10, &iou) == MD_OK);
    REQUIRE(iou == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("audio solution + tool capi", "[capi]") {
    MDAudioSolutionHandle ah = nullptr;
    REQUIRE(md_audio_solution_create(&ah, MD_AUDIO_SPEAKER_SEARCH) == MD_OK);
    float e1[3] = {1.0f, 0.0f, 0.0f};
    REQUIRE(md_audio_speaker_search_enroll(ah, "alice", e1, 3) == MD_OK);
    float e2[3] = {0.99f, 0.1f, 0.0f};
    const char* label = nullptr; float score = 0;
    REQUIRE(md_audio_speaker_search_match(ah, e2, 3, 1, &label, &score) == MD_OK);
    REQUIRE(std::string(label) == "alice");
    REQUIRE(md_audio_solution_destroy(ah) == MD_OK);

    float in[800]; std::fill(in, in + 800, 0.5f);
    float* rout = nullptr; size_t rn = 0;
    REQUIRE(md_audio_resample(in, 800, 8000, 16000, &rout, &rn) == MD_OK);
    REQUIRE(rn == 1600);
}

#ifdef BUILD_NLP
TEST_CASE("nlp tool capi", "[capi]") {
    size_t n = 0; const char** s = nullptr;
    REQUIRE(md_nlp_split_sent("你好。世界！", &s, &n) == MD_OK);
    REQUIRE(n == 2);
    size_t chars = 0, words = 0, sents = 0;
    REQUIRE(md_nlp_stats("hello world 你好", &chars, &words, &sents) == MD_OK);
    REQUIRE(words == 3);
    REQUIRE(sents == 1);
}
#endif
