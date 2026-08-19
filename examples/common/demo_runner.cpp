#include "demo_runner.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "csrc/vision.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"

namespace demo {
namespace {
constexpr const char* kFont = "../../test_data/msyh.ttc";

modeldeploy::RuntimeOption make_option(Backend b) {
    modeldeploy::RuntimeOption opt;
    switch (b) {
        case Backend::OrtCpu:
            opt.use_ort_backend();
            opt.use_cpu();
            opt.set_cpu_thread_num(4);
            break;
        case Backend::OrtGpuCudaEp:
            opt.use_ort_backend();
            opt.use_gpu(0);
            break;
        case Backend::OrtGpuTrtEp:
            opt.use_ort_backend();
            opt.use_gpu(0);
            opt.enable_trt = true;
            opt.enable_fp16 = true;
            opt.ort_option.trt_engine_cache_path = "./trt_engine";
            break;
        case Backend::MnnCpu:
            opt.use_mnn_backend();
            opt.use_cpu();
            break;
        case Backend::MnnCuda:
            opt.use_mnn_backend();
            opt.use_gpu(0);
            break;
        case Backend::MnnOpencl:
            opt.use_mnn_backend();
            opt.use_opencl(0);
            break;
        case Backend::MnnVulkan:
            opt.use_mnn_backend();
            opt.mnn_option.forward_type = modeldeploy::mnn::MNN_FORWARD_VULKAN;
            break;
        case Backend::SophgoF16:
        case Backend::SophgoInt8:
            opt.use_sophgo_backend(0);
            break;
    }
    return opt;
}

const char* backend_tag(Backend b) {
    switch (b) {
        case Backend::OrtCpu: return "ort_cpu";
        case Backend::OrtGpuCudaEp: return "ort_gpu_cuda_ep";
        case Backend::OrtGpuTrtEp: return "ort_gpu_trt_ep";
        case Backend::MnnCpu: return "mnn_cpu";
        case Backend::MnnCuda: return "mnn_cuda";
        case Backend::MnnOpencl: return "mnn_opencl";
        case Backend::MnnVulkan: return "mnn_vulkan";
        case Backend::SophgoF16: return "sophgo_tpu_f16";
        case Backend::SophgoInt8: return "sophgo_tpu_int8";
    }
    return "?";
}

// MNN 模型多数尚未提供 .mnn；未提供时仍指向 onnx 路径（运行时报文件/加载失败不会崩溃，便于外部对照）。
std::string model_path(const std::string& onnx, const std::string& bmodel, Backend b) {
    if (b == Backend::SophgoF16 || b == Backend::SophgoInt8)
        return "../../test_data/test_models/sophgo/" + bmodel;
    return "../../test_data/test_models/onnx/" + onnx;
}

template <typename Model, typename Result>
void bench(Model& m, const modeldeploy::vision::ImageData& im, Result* res,
           int warmup, int loop) {
    for (int i = 0; i < warmup; ++i) {
        m.predict(im, res, static_cast<TimerArray*>(nullptr));
    }
    TimerArray timers;
    for (int i = 0; i < loop; ++i) {
        m.predict(im, res, &timers);
    }
    timers.print_benchmark();
}
}  // namespace

int run_detection(Backend b) {
    std::string model = model_path("yolo26n/yolo26n.onnx", "zhgd_without_nms_640_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b);
    auto det = std::make_unique<modeldeploy::vision::detection::UltralyticsDet>(model, opt);
    if (!det->is_initialized()) {
        std::fprintf(stderr, "[%s] init failed: %s\n", backend_tag(b), model.c_str());
        return 1;
    }
    det->get_preprocessor().set_size({640, 640});
    const auto label_map = det->get_label_map("names");
    auto im = modeldeploy::vision::ImageData::imread(
        "../../test_data/test_images/test_pedestrian_attribute_scale.png");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    std::vector<modeldeploy::vision::DetectionResult> res;
    bench(*det, im, &res, 10, 50);
    modeldeploy::vision::dis_det(res);
    auto vis = modeldeploy::vision::vis_det(im, res, 0.5, label_map, kFont, 12, 0.3, false);
    (void)vis.imwrite("result_detection_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu objects\n", backend_tag(b), res.size());
    return 0;
}

int run_classification(Backend b) {
    std::string model = model_path("zhgd_ml.onnx", "zhgd_ml_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::classification::Classification>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    m->get_preprocessor().set_size({192, 256});
    m->get_preprocessor().disable_center_crop();
    m->get_postprocessor().set_multi_label(true);
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_face.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    modeldeploy::vision::ClassifyResult res;
    for (int i = 0; i < 10; ++i) {
        m->predict(im, &res);
    }
    for (int i = 0; i < 50; ++i) {
        m->predict(im, &res);
    }
    modeldeploy::vision::dis_cls(res);
    auto vis = modeldeploy::vision::vis_cls(im, res, 5, 0.5, kFont, 12, 0.3, false);
    (void)vis.imwrite("result_classification_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, label=%d score=%.4f\n", backend_tag(b),
                res.label_ids.empty() ? -1 : res.label_ids[0],
                res.scores.empty() ? -1.f : res.scores[0]);
    return 0;
}

int run_pose(Backend b) {
    std::string model = model_path("yolo11n/yolo11n-pose.onnx", "yolo26n/yolo26n-pose_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsPose>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    m->get_postprocessor().set_keypoints_num(17);
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_person.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    std::vector<modeldeploy::vision::KeyPointsResult> res;
    bench(*m, im, &res, 20, 80);
    auto vis = modeldeploy::vision::vis_pose(im, res, kFont, 12, 4, 0.3, false);
    (void)vis.imwrite("result_pose_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu persons\n", backend_tag(b), res.size());
    return 0;
}

int run_obb(Backend b) {
    std::string model = model_path("yolo11n/yolo11n-obb_nms.onnx", "yolo26n/yolo26n-obb_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsObb>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_obb1.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    std::vector<modeldeploy::vision::ObbResult> res;
    bench(*m, im, &res, 10, 100);
    auto vis = modeldeploy::vision::vis_obb(im, res, 0.2, kFont, 12, 0.3, 0);
    (void)vis.imwrite("result_obb_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu obbs\n", backend_tag(b), res.size());
    return 0;
}

int run_instance_seg(Backend b) {
    std::string model = model_path("yolo11n/yolo11n-seg_nms.onnx", "yolo26n/yolo26n-seg_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsSeg>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_person.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    std::vector<modeldeploy::vision::InstanceSegResult> res;
    bench(*m, im, &res, 10, 100);
    auto vis = modeldeploy::vision::vis_iseg(im, res, 0.2, kFont, 14, 0.5, false);
    (void)vis.imwrite("result_instance_seg_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu masks\n", backend_tag(b), res.size());
    return 0;
}

int run_sem(Backend b) {
    std::string model = model_path("yolo26n/yolo26n-sem.onnx", "yolo26n/yolo26n-sem_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsSem>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    const auto label_map = m->get_label_map("names");
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_sem_540.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    modeldeploy::vision::SemSegResult res;
    bench(*m, im, &res, 20, 100);
    auto vis = modeldeploy::vision::vis_sem(im, res, label_map, 0.5, true);
    (void)vis.imwrite("result_sem_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done %zux%zu\n", backend_tag(b),
                res.shape.empty() ? 0 : static_cast<size_t>(res.shape[0]),
                res.shape.size() < 2 ? 0 : static_cast<size_t>(res.shape[1]));
    return 0;
}

int run_depth(Backend b) {
    std::string model = model_path("yolo26n/yolo26n-depth.onnx", "yolo26n/yolo26n-depth_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsDepth>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_depth_540.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    modeldeploy::vision::DepthResult res;
    bench(*m, im, &res, 20, 100);
    auto vis = modeldeploy::vision::vis_depth(im, res, true, false);
    (void)vis.imwrite("result_depth_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done %zux%zu\n", backend_tag(b),
                res.shape.empty() ? 0 : static_cast<size_t>(res.shape[0]),
                res.shape.size() < 2 ? 0 : static_cast<size_t>(res.shape[1]));
    return 0;
}

int run_face_det(Backend b) {
    std::string model = model_path("face/scrfd_2.5g_bnkps_shape640x640.onnx", "face/scrfd_2.5g_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::face::Scrfd>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_face_detection4.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    auto im_bak = im.clone();
    std::vector<modeldeploy::vision::KeyPointsResult> res;
    bench(*m, im, &res, 10, 50);
    modeldeploy::vision::dis_lmk(res);
    auto vis = modeldeploy::vision::vis_keypoints(im_bak, res, kFont, 14, 2, 0.3, false, true);
    (void)vis.imwrite("result_face_det_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu faces\n", backend_tag(b), res.size());
    return 0;
}

int run_lpr_pipeline(Backend b) {
    std::string det = "../../test_data/test_models/onnx/yolov5plate.onnx";
    std::string rec = "../../test_data/test_models/onnx/plate_recognition_color.onnx";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::lpr::LprPipeline>(det, rec, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_lpr_detection.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    auto im_bak = im.clone();
    std::vector<modeldeploy::vision::LprResult> res;
    bench(*m, im, &res, 5, 20);
    auto vis = modeldeploy::vision::vis_lpr(im_bak, res, kFont);
    (void)vis.imwrite("result_lpr_pipeline_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu plates\n", backend_tag(b), res.size());
    return 0;
}

int run_ocr_pipeline(Backend b) {
    const char* det = "../../test_data/test_models/onnx/ocr/ppocrv5_mobile/det_infer2.onnx";
    const char* cls = "../../test_data/test_models/onnx/ocr/ppocrv4_mobile/cls_infer.onnx";
    const char* rec = "../../test_data/test_models/onnx/ocr/ppocrv5_mobile/rec_infer1.onnx";
    const char* dict = "../../test_data/dict.txt";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::ocr::PaddleOCR>(det, cls, rec, dict, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    m->set_rec_batch_size(8);
    m->get_detector()->get_preprocessor().set_max_side_len(1440);
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/ocr2.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    modeldeploy::vision::OCRResult res;
    bench(*m, im, &res, 5, 20);
    modeldeploy::vision::dis_ocr(res);
    auto vis = modeldeploy::vision::vis_ocr(im, res, kFont);
    (void)vis.imwrite("result_ocr_pipeline_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu boxes\n", backend_tag(b), res.boxes.size());
    return 0;
}

int run_pedestrian_attribute(Backend b) {
    std::string det = "../../test_data/test_models/onnx/zhgd_det.onnx";
    std::string ml = "../../test_data/test_models/onnx/zhgd_ml.onnx";
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::pipeline::PedestrianAttribute>(det, ml, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    m->set_cls_batch_size(8);
    m->set_det_input_size({1280, 1280});
    m->set_det_threshold(0.5);
    m->set_cls_input_size({192, 256});
    auto im = modeldeploy::vision::ImageData::imread(
        "../../test_data/test_images/test_pedestrian_attribute_scale.png");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    std::vector<modeldeploy::vision::AttributeResult> res;
    bench(*m, im, &res, 10, 50);
    modeldeploy::vision::dis_attr(res);
    std::unordered_map<int, std::string> label_map;
    label_map.insert({0, "safety_helmet"});
    label_map.insert({1, "reflective_vest"});
    label_map.insert({2, "safety_rope"});
    label_map.insert({3, "work_uniform"});
    auto vis = modeldeploy::vision::vis_attr(im, res, 0.5, label_map, kFont, 6, 0.15, false, {0, 1});
    (void)vis.imwrite("result_pedestrian_attribute_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu persons\n", backend_tag(b), res.size());
    return 0;
}
}  // namespace demo
