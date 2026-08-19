#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "csrc/vision.h"
#ifdef WITH_GPU
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#endif

namespace fs = std::filesystem;
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::detection;
using namespace modeldeploy::vision::classification;
using namespace modeldeploy::vision::face;
using namespace modeldeploy::vision::ocr;

static fs::path get_test_data() {
    const char* env = std::getenv("TEST_DATA_DIR");
    if (env && *env) return fs::path(env) / "test_data";
    return fs::current_path() / "test_data";
}

static fs::path model_path(const std::string& rel) {
    return get_test_data() / "test_models" / rel;
}

static fs::path image_path(const std::string& name) {
    return get_test_data() / "test_images" / name;
}

static ImageData load_image(const std::string& name) {
    return ImageData::imread(image_path(name).string());
}

template<typename Model, typename Result>
static void test_model_predict(Model& model, const ImageData& img, std::vector<Result>* results) {
    REQUIRE(model.predict(img, results));
    REQUIRE_FALSE(results->empty());
}

// ==================== Classification ====================
TEST_CASE("Classification model", "[vision_models]") {
    auto modelfile = model_path("onnx/yolo26n/yolo26n-cls.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    Classification model(modelfile.string(), opt);
    REQUIRE(model.name() == "Classification");

    auto img = load_image("test_person.jpg");
    if (img.empty()) return;

    ClassifyResult result;
    REQUIRE(model.predict(img, &result));
    REQUIRE(result.label_ids.size() > 0);
    REQUIRE(result.scores.size() > 0);
    REQUIRE(result.label_ids[0] >= 0);
    REQUIRE(result.scores[0] > 0);

    auto& preproc = model.get_preprocessor();
    auto& postproc = model.get_postprocessor();
}

// ==================== Ultralytics Detection ====================
TEST_CASE("UltralyticsDet model", "[vision_models]") {
    auto modelfile = model_path("onnx/yolo26n/yolo26n.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    UltralyticsDet model(modelfile.string(), opt);
    REQUIRE(model.name() == "UltralyticsDet");

    auto img = load_image("test_detection0.jpg");
    if (img.empty()) return;

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    REQUIRE(results.size() > 0);
    for (auto& r : results) {
        REQUIRE(r.box.width > 0);
        REQUIRE(r.box.height >= 0);
        REQUIRE(r.label_id >= 0);
        REQUIRE(r.score > 0);
    }
}

// yolo26n_b8.engine（TRT，动态 batch，end2end NMS 输出 [batch,300,6]）：
// 单图 + batch_predict(batch=3) 均须能加载并产出检测
TEST_CASE("UltralyticsDet yolo26n TRT engine (dynamic batch)", "[vision_models][gpu][trt]") {
    auto modelfile = model_path("trt/yolo26n_b8.engine");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_gpu(0);
    opt.use_trt_backend();

    UltralyticsDet model(modelfile.string(), opt);
    if (!model.is_initialized()) {
        std::cerr << "yolo26n_b8 TRT engine not initializable (TRT unavailable) — skip" << std::endl;
        return;
    }

    auto img = load_image("test_detection0.jpg");
    auto img2 = load_image("test_person.jpg");
    auto img3 = load_image("test_detection1.jpg");
    if (img.empty() || img2.empty() || img3.empty()) return;

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    std::cout << "[yolo26n_b8] single detections=" << results.size() << std::endl;
    REQUIRE(results.size() > 0);
    for (auto& r : results) {
        REQUIRE(r.box.width > 0);
        REQUIRE(r.box.height >= 0);
        REQUIRE(r.label_id >= 0);
        REQUIRE(r.score > 0);
    }

    std::vector<std::vector<DetectionResult>> batched;
    REQUIRE(model.batch_predict({img, img2, img3}, &batched, nullptr));
    REQUIRE(batched.size() == 3);
    std::cout << "[yolo26n_b8] batch sizes=" << batched[0].size() << ","
              << batched[1].size() << "," << batched[2].size() << std::endl;
    for (auto& r : batched[0]) {
        REQUIRE(r.box.width > 0);
        REQUIRE(r.box.height >= 0);
        REQUIRE(r.label_id >= 0);
        REQUIRE(r.score > 0);
    }
}

// 同一 NV12 缓冲：predict(ImageData) 的 NV12 分叉是统一单入口，须能零拷贝预处并产出合理结果
// [model]：需模型文件，缺文件时跳过（不自红）
TEST_CASE("UltralyticsDet predict(ImageData) on NV12 device frame", "[model]") {
    auto modelfile = model_path("onnx/yolo26n/yolo26n.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    UltralyticsDet model(modelfile.string(), opt);

    auto img = load_image("test_detection0.jpg");
    if (img.empty()) return;
    const int w = img.width(), h = img.height();
    const int step_src = img.plane(0).step > 0 ? img.plane(0).step : w * 3;

    // BGR → NV12：Y 取 luma（保留场景结构），UV 置中性灰 128
    std::vector<uint8_t> y(static_cast<size_t>(w) * h);
    for (int i = 0; i < h; ++i) {
        const uint8_t* row = img.plane(0).data + static_cast<size_t>(i) * step_src;
        for (int j = 0; j < w; ++j) {
            const uint8_t b = row[j * 3 + 0], g = row[j * 3 + 1], r = row[j * 3 + 2];
            y[static_cast<size_t>(i) * w + j] =
                static_cast<uint8_t>((77 * r + 150 * g + 29 * b + 128) >> 8);
        }
    }
    std::vector<uint8_t> uv(static_cast<size_t>(w) * h / 2, 128);

    ImageData::Plane pl[2] = {{y.data(), w}, {uv.data(), w}};
    ImageData frame = ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, modeldeploy::Device::CPU);
    REQUIRE(frame.format() == MdImageType::NV12);
    REQUIRE(frame.plane_count() == 2);

    std::vector<DetectionResult> r_predict;
    REQUIRE(model.predict(frame, &r_predict, nullptr));
    REQUIRE(r_predict.size() > 0);
    for (auto& r : r_predict) {
        REQUIRE(r.box.width > 0);
        REQUIRE(r.box.height >= 0);
        REQUIRE(r.label_id >= 0);
        REQUIRE(r.score > 0);
    }
}

#ifdef WITH_GPU
namespace {
    // BGR packed ImageData -> 真实 NV12 主机缓冲（Y + 交织 UV），供上传到 GPU
    void bgr_to_nv12_host(const ImageData& bgr,
                          std::vector<uint8_t>* y_out, std::vector<uint8_t>* uv_out) {
        const int w = bgr.width(), h = bgr.height();
        const int step_src = bgr.plane(0).step > 0 ? bgr.plane(0).step : w * 3;
        cv::Mat bgr_mat(h, w, CV_8UC3, const_cast<uint8_t*>(bgr.plane(0).data), step_src);
        cv::Mat i420;
        cv::cvtColor(bgr_mat, i420, cv::COLOR_BGR2YUV_I420);
        const uint8_t* p = i420.ptr<uint8_t>();
        y_out->resize(static_cast<size_t>(w) * h);
        uv_out->resize(static_cast<size_t>(w) * h / 2);
        std::memcpy(y_out->data(), p, static_cast<size_t>(w) * h);
        const uint8_t* up = p + static_cast<size_t>(w) * h;
        const uint8_t* vp = p + static_cast<size_t>(w) * h + static_cast<size_t>(w) * h / 4;
        const size_t n = static_cast<size_t>(w) * h / 4;
        for (size_t i = 0; i < n; ++i) {
            (*uv_out)[2 * i] = up[i];
            (*uv_out)[2 * i + 1] = vp[i];
        }
    }

    // 排序后取最高分检测
    static void top_detection(std::vector<DetectionResult>* rs, DetectionResult* out) {
        REQUIRE_FALSE(rs->empty());
        std::sort(rs->begin(), rs->end(), [](const DetectionResult& a, const DetectionResult& b) {
            return a.score > b.score;
        });
        *out = rs->front();
    }

    static float box_iou(const Rect2f& a, const Rect2f& b) {
        const float ax2 = a.x + a.width, ay2 = a.y + a.height;
        const float bx2 = b.x + b.width, by2 = b.y + b.height;
        const float ix = std::max(0.0f, std::min(ax2, bx2) - std::max(a.x, b.x));
        const float iy = std::max(0.0f, std::min(ay2, by2) - std::max(a.y, b.y));
        const float inter = ix * iy;
        const float ua = a.width * a.height + b.width * b.height - inter;
        return ua > 0 ? inter / ua : 0.0f;
    }
} // namespace

// 端到端：真实 NV12 帧常驻 GPU 显存 → ImageData(Device::GPU) → predict() 全程零拷贝
// CUDA 预处理 kernel 直接读 plane(0)/plane(1) 设备指针 → GPU Tensor → ORT CUDA EP 推理
TEST_CASE("UltralyticsDet predict(ImageData) on GPU NV12 device frame (zero-copy e2e)", "[model][gpu]") {
    auto modelfile = model_path("onnx/yolo26n/yolo26n.onnx");
    if (!fs::exists(modelfile)) return;

    auto img = load_image("bus.jpg");
    if (img.empty()) return;
    const int w = img.width(), h = img.height();

    // 1) 主机 BGR -> NV12（真实色度，非中性灰）
    std::vector<uint8_t> h_y, h_uv;
    bgr_to_nv12_host(img, &h_y, &h_uv);

    // 2) 上传 Y/UV 到 GPU 显存（真实 NV12 frame 常驻 device memory）
    uint8_t* d_y = nullptr;
    uint8_t* d_uv = nullptr;
    REQUIRE(cudaMalloc(&d_y, static_cast<size_t>(w) * h) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_uv, static_cast<size_t>(w) * h / 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_y, h_y.data(), static_cast<size_t>(w) * h, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_uv, h_uv.data(), static_cast<size_t>(w) * h / 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    // owner 维护 cudaMalloc'd 缓冲（ImageData 借用，随 owner 析构释放）
    std::shared_ptr<void> owner(static_cast<void*>(nullptr),
                                [d_y, d_uv](void*) {
                                    if (d_y) cudaFree(d_y);
                                    if (d_uv) cudaFree(d_uv);
                                });
    ImageData::Plane pl[2] = {{d_y, w}, {d_uv, w}};
    ImageData frame = ImageData::from_planes(pl, 2, MdImageType::NV12, w, h,
                                             modeldeploy::Device::GPU, owner);

    // 3) 验证帧是 GPU 且 plane 指向设备内存（零拷贝，未经 from_planes 复制）
    REQUIRE(frame.device() == modeldeploy::Device::GPU);
    REQUIRE(frame.plane_count() == 2);
    REQUIRE(frame.plane(0).data == d_y);
    REQUIRE(frame.plane(1).data == d_uv);

    // 4) 完整 model.predict()：CUDA 预处理 kernel 直读设备 plane + ORT CUDA EP 推理
    modeldeploy::RuntimeOption opt;
    opt.use_gpu(0);
    UltralyticsDet model(modelfile.string(), opt);

    std::vector<DetectionResult> r_gpu;
    REQUIRE(model.predict(frame, &r_gpu, nullptr));
    REQUIRE_FALSE(r_gpu.empty());
    for (auto& r : r_gpu) {
        REQUIRE(r.box.width > 0);
        REQUIRE(r.box.height >= 0);
        REQUIRE(r.label_id >= 0);
        REQUIRE(r.score > 0);
    }

    // 5) 交叉校验：同图 CPU predict（BGR 打包帧）→ 最高分检测框应与设备路径 IoU 重叠
    modeldeploy::RuntimeOption copt;
    copt.use_cpu();
    UltralyticsDet cmodel(modelfile.string(), copt);
    auto cpu_img = load_image("bus.jpg");
    REQUIRE_FALSE(cpu_img.empty());
    std::vector<DetectionResult> r_cpu;
    REQUIRE(cmodel.predict(cpu_img, &r_cpu, nullptr));
    REQUIRE_FALSE(r_cpu.empty());

    DetectionResult top_gpu, top_cpu;
    top_detection(&r_gpu, &top_gpu);
    top_detection(&r_cpu, &top_cpu);
    INFO("GPU top box=(" << top_gpu.box.x << "," << top_gpu.box.y << "," <<
        top_gpu.box.width << "," << top_gpu.box.height << ") score=" << top_gpu.score
        << " | CPU top box=(" << top_cpu.box.x << "," << top_cpu.box.y << "," <<
        top_cpu.box.width << "," << top_cpu.box.height << ") score=" << top_cpu.score);
    REQUIRE(box_iou(top_gpu.box, top_cpu.box) > 0.5f);
}

// 模型 Clone 真共享验证（ORT/GPU）：克隆必须复用已加载的 ORT session（共享显存/权重），
// 而非重新加载模型。用 cudaMemGetInfo 实测：克隆造成的显存增量应远小于独立再加载一份模型。
TEST_CASE("ORT GPU model clone shares device memory (no re-load)", "[model][gpu]") {
    auto modelfile = model_path("onnx/yolo26n/yolo26n.onnx");
    if (!fs::exists(modelfile)) return;
    auto img = load_image("bus.jpg");
    if (img.empty()) return;

    modeldeploy::RuntimeOption opt;
    opt.use_gpu(0);
    UltralyticsDet model(modelfile.string(), opt);

    size_t total = 0, free0 = 0;
    REQUIRE(cudaMemGetInfo(&free0, &total) == cudaSuccess);

    // 克隆应共享已加载的 ORT session（显存/权重），不重载模型
    auto clone = model.clone();
    REQUIRE(clone != nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    size_t free1 = 0;
    REQUIRE(cudaMemGetInfo(&free1, &total) == cudaSuccess);
    const size_t used_clone = (free0 > free1) ? (free0 - free1) : 0;

    // 原模型与克隆各自 predict 结果一致
    std::vector<DetectionResult> r0, r1;
    REQUIRE(model.predict(img, &r0, nullptr));
    REQUIRE(clone->predict(img, &r1, nullptr));
    REQUIRE_FALSE(r0.empty());
    REQUIRE_FALSE(r1.empty());
    DetectionResult t0, t1;
    top_detection(&r0, &t0);
    top_detection(&r1, &t1);
    REQUIRE(box_iou(t0.box, t1.box) > 0.5f);

    // 对照：独立再加载一份模型（新 session）必然为其单独分配权重显存
    UltralyticsDet model2(modelfile.string(), opt);
    size_t free2 = 0;
    REQUIRE(cudaMemGetInfo(&free2, &total) == cudaSuccess);
    const size_t used_full_load = (free1 > free2) ? (free1 - free2) : 0;

    INFO("clone GPU delta=" << used_clone << " bytes | independent full load delta="
        << used_full_load << " bytes");
    // 克隆 Δ 应远小于一份完整模型加载（< 1/4），证明是共享显存而非重载复制权重
    REQUIRE(used_full_load > 0);
    REQUIRE(used_clone < used_full_load / 4);
}
#endif // WITH_GPU

// ==================== Ultralytics Segmentation ====================
TEST_CASE("UltralyticsSeg model", "[vision_models]") {
    auto modelfile = model_path("onnx/yolo26n/yolo26n-seg.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    UltralyticsSeg model(modelfile.string(), opt);
    REQUIRE(model.name() == "UltralyticsSeg");

    auto img = load_image("test_person.jpg");
    if (img.empty()) return;

    std::vector<InstanceSegResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    REQUIRE(results.size() > 0);
    for (auto& r : results) {
        REQUIRE(r.box.width > 0);
        REQUIRE(r.label_id >= 0);
        REQUIRE(r.score > 0);
    }
}

// ==================== Ultralytics Pose ====================
TEST_CASE("UltralyticsPose model", "[vision_models]") {
    auto modelfile = model_path("onnx/yolo26n/yolo26n-pose.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    UltralyticsPose model(modelfile.string(), opt);
    REQUIRE(model.name() == "UltralyticsPose");

    auto img = load_image("test_person.jpg");
    if (img.empty()) return;

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    REQUIRE(results.size() > 0);
    for (auto& r : results) {
        REQUIRE(r.box.width > 0);
        REQUIRE(r.keypoints.size() > 0);
    }
}

// ==================== Ultralytics OBB ====================
TEST_CASE("UltralyticsObb model", "[vision_models]") {
    auto modelfile = model_path("onnx/yolo26n/yolo26n-obb.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    UltralyticsObb model(modelfile.string(), opt);
    REQUIRE(model.name() == "UltralyticsObb");

    auto img = load_image("test_obb.jpg");
    if (img.empty()) {
        img = load_image("test_detection0.jpg");
    }
    if (img.empty()) return;

    std::vector<ObbResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    REQUIRE(results.size() > 0);
    for (auto& r : results) {
        REQUIRE(r.rotated_box.xc > 0);
        REQUIRE(r.label_id >= 0);
        REQUIRE(r.score > 0);
    }
}

// ==================== Batch Predict ====================
TEST_CASE("Batch predict for vision models", "[vision_models]") {
    auto modelfile = model_path("onnx/yolo26n/yolo26n.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    UltralyticsDet model(modelfile.string(), opt);

    auto img1 = load_image("test_detection0.jpg");
    auto img2 = load_image("test_person.jpg");
    if (img1.empty() || img2.empty()) return;

    std::vector<std::vector<DetectionResult>> results;
    REQUIRE(model.batch_predict({img1, img2}, &results, nullptr));
    REQUIRE(results.size() == 2);
    REQUIRE(results[0].size() > 0);
    REQUIRE(results[1].size() > 0);
}

// ==================== Face Models ====================
TEST_CASE("Scrfd face detection model", "[vision_models]") {
    auto modelfile = model_path("onnx/face/scrfd_2.5g_bnkps_shape640x640.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    Scrfd model(modelfile.string(), opt);

    auto img = load_image("test_face_detection.jpg");
    if (img.empty()) return;

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    REQUIRE(results.size() > 0);
}

TEST_CASE("SeetaFaceAge model", "[vision_models]") {
    auto modelfile = model_path("onnx/face/age_predictor.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    SeetaFaceAge model(modelfile.string(), opt);

    auto img = load_image("test_face.jpg");
    if (img.empty()) return;

    int age = -1;
    REQUIRE(model.predict(img, &age));
    REQUIRE(age >= 0);
}

TEST_CASE("SeetaFaceGender model", "[vision_models]") {
    auto modelfile = model_path("onnx/face/gender_predictor.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    SeetaFaceGender model(modelfile.string(), opt);

    auto img = load_image("test_face_gender.jpg");
    if (img.empty()) return;

    int gender = -1;
    REQUIRE(model.predict(img, &gender));
    REQUIRE(gender >= 0);
}

// ==================== OCR Models ====================
TEST_CASE("OCR DBDetector model", "[vision_models]") {
    auto modelfile = model_path("onnx/ocr/ppocrv6_tiny/det_infer.onnx");
    if (!fs::exists(modelfile)) {
        modelfile = model_path("onnx/ocr/ppocrv5_mobile/det_infer.onnx");
    }
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    DBDetector model(modelfile.string(), opt);
    REQUIRE(model.name() == "ppocr/ocr_det");

    auto img = load_image("test_ocr.png");
    if (img.empty()) return;

    std::vector<std::array<int, 8>> boxes;
    REQUIRE(model.predict(img, &boxes, nullptr));
    REQUIRE(boxes.size() > 0);
}

TEST_CASE("OCR Classifier model", "[vision_models]") {
    auto modelfile = model_path("onnx/ocr/ppocrv6_tiny/cls_infer.onnx");
    if (!fs::exists(modelfile)) {
        modelfile = model_path("onnx/ocr/ppocrv5_mobile/cls_infer.onnx");
    }
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    Classifier model(modelfile.string(), opt);

    auto img = load_image("test_ocr.png");
    if (img.empty()) return;

    int32_t cls_label = -1;
    float cls_score = 0;
    REQUIRE(model.predict(img, &cls_label, &cls_score));
    REQUIRE(cls_label >= 0);
}

TEST_CASE("OCR Recognizer model", "[vision_models]") {
    auto modelfile = model_path("onnx/ocr/ppocrv6_tiny/rec_infer.onnx");
    if (!fs::exists(modelfile)) {
        modelfile = model_path("onnx/ocr/ppocrv5_mobile/rec_infer.onnx");
    }

    auto dict = get_test_data() / "ppocrv6_tiny_dict.txt";
    if (!fs::exists(dict)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    Recognizer model(modelfile.string(), dict.string(), opt);

    // rec 模型输入应为单行文本裁剪图；test_ocr.png 是多行文档图（v6 rec 返回空文本）。
    auto img = load_image("test_ocr_recognition.jpg");
    if (img.empty()) img = load_image("test_ocr.png");
    if (img.empty()) return;

    std::string text;
    float score = 0;
    REQUIRE(model.predict(img, &text, &score, nullptr));
    REQUIRE_FALSE(text.empty());
    REQUIRE(score > 0);
}

// ==================== Preprocessor access ====================
TEST_CASE("Preprocessor/Postprocessor access", "[vision_models]") {
    auto modelfile = model_path("onnx/yolo26n/yolo26n.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    UltralyticsDet model(modelfile.string(), opt);
    auto& preproc = model.get_preprocessor();
    auto& postproc = model.get_postprocessor();

    preproc.set_size({640, 640});
    auto size = preproc.get_size();
    REQUIRE(size.size() == 2);
}
