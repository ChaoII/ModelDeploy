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
#include "test_gpu_utils.h"

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
    MD_TEST_GPU_OR_SKIP();
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
    // NV12 有限色域（bgr_to_nv12_host 经 OpenCV I420，Y 16~235）在反向 BT.601 重建时
    // 产生约 0.02（≈6 灰阶）系统色偏，足以让低分差类别间的 top-1 分数排序翻转
    // （实测两路径检出框集完全一致，仅得分序不同）。故用集合级 IoU 匹配而非单比 top-1。
    int matched_g2c = 0, matched_c2g = 0;
    for (const auto& g : r_gpu) {
        for (const auto& c : r_cpu) {
            if (box_iou(g.box, c.box) > 0.5f) { ++matched_g2c; break; }
        }
    }
    for (const auto& c : r_cpu) {
        for (const auto& g : r_gpu) {
            if (box_iou(c.box, g.box) > 0.5f) { ++matched_c2g; break; }
        }
    }
    INFO("cross-set matches: GPU->CPU " << matched_g2c << "/" << r_gpu.size()
         << " CPU->GPU " << matched_c2g << "/" << r_cpu.size());
    REQUIRE(matched_g2c >= 3);
    REQUIRE(matched_c2g >= 3);
}

// 模型 Clone 真共享验证（ORT/GPU）：克隆必须复用已加载的 ORT session（共享显存/权重），
// 而非重新加载模型。用 cudaMemGetInfo 实测：克隆造成的显存增量应远小于独立再加载一份模型。
TEST_CASE("ORT GPU model clone shares device memory (no re-load)", "[model][gpu]") {
    MD_TEST_GPU_OR_SKIP();
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
    model.get_preprocessor().set_size({1024, 1024});

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
    auto modelfile = model_path("onnx/yolo11n/yolo11n.onnx");  // 动态 batch 模型,支持 batch_predict
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
    auto modelfile = model_path("onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx");
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
    auto modelfile = model_path("onnx/seetaface/age_predictor.onnx");
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
    auto modelfile = model_path("onnx/seetaface/gender_predictor.onnx");
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

#ifdef WITH_GPU
TEST_CASE("DEBUG NV12 preproc A/B/C compare", "[debug][gpu]") {
    MD_TEST_GPU_OR_SKIP();
    auto img = load_image("bus.jpg");
    if (img.empty()) return;
    const int w = img.width(), h = img.height();
    CAPTURE(w, h);

    std::vector<uint8_t> h_y, h_uv;
    bgr_to_nv12_host(img, &h_y, &h_uv);

    uint8_t* d_y = nullptr;
    uint8_t* d_uv = nullptr;
    REQUIRE(cudaMalloc(&d_y, static_cast<size_t>(w) * h) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_uv, static_cast<size_t>(w) * h / 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_y, h_y.data(), static_cast<size_t>(w) * h, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_uv, h_uv.data(), static_cast<size_t>(w) * h / 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::shared_ptr<void> owner(static_cast<void*>(nullptr), [d_y, d_uv](void*) { if (d_y) cudaFree(d_y); if (d_uv) cudaFree(d_uv); });

    std::shared_ptr<VisionProcessorBackend> gpu_backend = create_processor_backend(modeldeploy::Device::GPU, modeldeploy::Backend::ORT, 0);
    UltralyticsPreprocessor pg;
    pg.set_processor_backend(gpu_backend);

    modeldeploy::Tensor tA; LetterBoxRecord rA;
    REQUIRE(pg.run(h_y.data(), h_uv.data(), {w, h}, w, w, &tA, &rA, modeldeploy::Device::CPU));
    modeldeploy::Tensor tB; LetterBoxRecord rB;
    REQUIRE(pg.run(d_y, d_uv, {w, h}, w, w, &tB, &rB, modeldeploy::Device::GPU));

    std::shared_ptr<VisionProcessorBackend> cpu_backend = create_processor_backend(modeldeploy::Device::CPU, modeldeploy::Backend::ORT, 0);
    UltralyticsPreprocessor pc;
    pc.set_processor_backend(cpu_backend);
    modeldeploy::Tensor tC; LetterBoxRecord rC;
    REQUIRE(pc.run(h_y.data(), h_uv.data(), {w, h}, w, w, &tC, &rC, modeldeploy::Device::CPU));

    auto read_gpu = [](modeldeploy::Tensor& t) {
        std::vector<float> out(t.size());
        cudaError_t e = cudaMemcpy(out.data(), t.data(), t.byte_size(), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        REQUIRE(e == cudaSuccess);
        return out;
    };
    std::vector<float> va = read_gpu(tA);
    std::vector<float> vb = read_gpu(tB);
    std::vector<float> vc(tC.size());
    std::memcpy(vc.data(), tC.data(), tC.byte_size());

    auto stats = [](const std::vector<float>& v, size_t n) {
        double m = 0, s = 0;
        for (size_t i = 0; i < v.size(); ++i) { m += v[i]; s += v[i] * v[i]; }
        m /= v.size(); s = std::sqrt(s / v.size() - m * m);
        return std::pair<double, double>{m, s};
    };
    auto ch_stats = [&](const std::vector<float>& v) {
        size_t n = v.size() / 3;
        auto s0 = stats(v, 0); // 整体均值,沿用
        return s0;
    };
    auto mad = [](const std::vector<float>& a, const std::vector<float>& b) {
        float m = 0;
        for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
        return m;
    };
    auto thumb = [](const std::vector<float>& v, size_t n) {
        // n x n 块均值,通道0
        size_t per = v.size() / 3;
        std::vector<float> out(n * n);
        const size_t dw = static_cast<size_t>(std::sqrt((double)per));
        for (size_t by = 0; by < n; ++by) for (size_t bx = 0; bx < n; ++bx) {
            double s = 0; size_t cnt = 0;
            for (size_t yy = by * dw / n; yy < (by + 1) * dw / n; ++yy)
                for (size_t xx = bx * dw / n; xx < (bx + 1) * dw / n; ++xx)
                    if (yy * dw + xx < per) { s += v[yy * dw + xx]; ++cnt; }
            out[by * n + bx] = static_cast<float>(s / std::max<size_t>(1, cnt));
        }
        return out;
    };

    std::cout << "DBG n = " << tA.size() / 3 << " (dst 640x640)" << std::endl;
    std::cout << "DBG A(host->gpu) mean/std ch0=" << ch_stats(va).first << "/" << ch_stats(va).second
              << " B(zerocopy)=" << ch_stats(vb).first << "/" << ch_stats(vb).second
              << " C(cpu)=" << ch_stats(vc).first << "/" << ch_stats(vc).second << std::endl;
    std::cout << "DBG mad A-vs-C=" << mad(va, vc) << "  B-vs-A=" << mad(vb, va)
              << "  B-vs-C=" << mad(vb, vc) << std::endl;
    std::cout << "DBG lbr A(scale,pad)=" << rA.scale << "," << rA.pad_w << "," << rA.pad_h
              << " B=" << rB.scale << "," << rB.pad_w << "," << rB.pad_h
              << " C=" << rC.scale << "," << rC.pad_w << "," << rC.pad_h << std::endl;

    // ---- 全链路 predict：host NV12 vs device NV12 vs CPU(BGR) ----
    auto print_top = [](const char* tag, const std::vector<DetectionResult>& v) {
        if (v.empty()) { std::cout << "DBG " << tag << " EMPTY" << std::endl; return; }
        auto srt = v;
        std::sort(srt.begin(), srt.end(),
                  [](const DetectionResult& a, const DetectionResult& b) { return a.score > b.score; });
        for (size_t i = 0; i < srt.size() && i < 4; ++i)
            std::cout << "DBG " << tag << " [" << i << "] cls=" << srt[i].label_id
                      << " box=(" << srt[i].box.x << "," << srt[i].box.y
                      << "," << srt[i].box.width << "," << srt[i].box.height
                      << ") score=" << srt[i].score << std::endl;
        if (srt.size() > 4) std::cout << "DBG " << tag << " ... total " << srt.size() << std::endl;
    };
    {
        auto modelfile = model_path("onnx/yolo26n/yolo26n.onnx");
        modeldeploy::RuntimeOption gopt; gopt.use_gpu(0);
        UltralyticsDet gmodel(modelfile.string(), gopt);
        modeldeploy::RuntimeOption copt; copt.use_cpu();
        UltralyticsDet cmodel(modelfile.string(), copt);

        // D1: host NV12 frame
        ImageData::Plane hpl[2] = {{h_y.data(), w}, {h_uv.data(), w}};
        ImageData frame_host = ImageData::from_planes(hpl, 2, MdImageType::NV12, w, h, modeldeploy::Device::CPU);
        std::vector<DetectionResult> d1;
        REQUIRE(gmodel.predict(frame_host, &d1, nullptr));
        print_top("D1 gpu-hostNV12", d1);

        // D2: device NV12 frame
        ImageData::Plane dpl[2] = {{d_y, w}, {d_uv, w}};
        ImageData frame_dev = ImageData::from_planes(dpl, 2, MdImageType::NV12, w, h, modeldeploy::Device::GPU, owner);
        std::vector<DetectionResult> d2;
        REQUIRE(gmodel.predict(frame_dev, &d2, nullptr));
        print_top("D2 gpu-devNV12", d2);

        std::vector<DetectionResult> d3;
        REQUIRE(cmodel.predict(img, &d3, nullptr));
        print_top("D3 cpu-BGR   ", d3);

        // 对比 NV12->640 与 BGR->640 张量是否一致（验证色彩转换）
        std::vector<modeldeploy::Tensor> outs;
        std::vector<LetterBoxRecord> recs;
        REQUIRE(pc.run({img}, &outs, &recs));
        const float* q = static_cast<const float*>(outs[0].data());
        std::vector<float> vE(q, q + outs[0].size());
        std::cout << "DBG BGR tensor: n=" << outs[0].size() / 3
                  << " lbr=" << recs[0].scale << "," << recs[0].pad_w << "," << recs[0].pad_h << std::endl;
        std::cout << "DBG mad NV12cpu-vs-BGRcpu=" << mad(vc, vE)
                  << "  mean/ch std NV12=" << ch_stats(vc).first << "/" << ch_stats(vc).second
                  << "  BGR=" << ch_stats(vE).first << "/" << ch_stats(vE).second << std::endl;
        // 按通道分别看 0/1/2 的均值差异（判断是否色乘性/通道错位）
        auto ca = [&](const std::vector<float>& v, int c) {
            double s = 0; size_t n = 0;
            for (size_t i = c; i < v.size(); i += 3) { s += v[i]; ++n; }
            return n ? s / n : 0.0;
        };
        std::cout << "DBG mean[Nv12] ch0/1/2=" << ca(vc, 0) << "/" << ca(vc, 1) << "/" << ca(vc, 2)
                  << "  mean[BGR]=" << ca(vE, 0) << "/" << ca(vE, 1) << "/" << ca(vE, 2) << std::endl;

        {   const char* d = "C:/Users/aichao/AppData/Local/Temp/opencode/";
            FILE* f = fopen((std::string(d) + "nv12.chw").c_str(), "wb");
            fwrite(vc.data(), sizeof(float), vc.size(), f); fclose(f);
            f = fopen((std::string(d) + "bgr.chw").c_str(), "wb");
            fwrite(vE.data(), sizeof(float), vE.size(), f); fclose(f);
            f = fopen((std::string(d) + "meta.txt").c_str(), "wt");
            fprintf(f, "plane=%zu n=%zu src_w=%d src_h=%d scale=%.6f pad_w=%.3f pad_h=%.3f dwh=640 640\n",
                    vc.size() / 3, vc.size(), w, h, rA.scale, rA.pad_w, rA.pad_h);
            fclose(f);
        }
        {
            const size_t plane = tA.size() / 3;
            double sum_v = 0, sum_p = 0; size_t cn = 0, pn = 0;
            for (size_t p = 0; p < vc.size(); ++p) {
                float dx = fabsf(vc[p] - vE[p]);
                size_t k = p % plane;
                int px = static_cast<int>(k % 640), py = static_cast<int>(k / 640);
                bool in_pad = (px + 0.5f - rA.pad_w) < 0 || (py + 0.5f - rA.pad_h) < 0 ||
                              (px - rA.pad_w) / rA.scale >= static_cast<float>(w) ||
                              (py - rA.pad_h) / rA.scale >= static_cast<float>(h);
                if (in_pad) { sum_p += dx; ++pn; } else { sum_v += dx; ++cn; }
            }
            std::cout << "DBG meanAbsDiff valid=" << (cn ? sum_v / cn : 0)
                      << " pad=" << (pn ? sum_p / pn : 0)
                      << " (validN=" << cn << " padN=" << pn << ")" << std::endl;
        }
    }
}
#endif
