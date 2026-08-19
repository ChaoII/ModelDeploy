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

// trt_profile 形如 "x:1x3x640x640"，仅用于 ORT TRT EP（ORT 在线建引擎需 min/opt/max）。
// 原生 TRT 后端(_trt)直接加载预构建 .engine，无需 profile。
modeldeploy::RuntimeOption make_option(Backend b, const char* trt_profile = nullptr) {
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
            if (trt_profile) {
                opt.set_trt_min_shape(trt_profile);
                opt.set_trt_opt_shape(trt_profile);
                opt.set_trt_max_shape(trt_profile);
            }
            break;
        case Backend::Trt:
            opt.use_trt_backend();
            opt.use_gpu(0);
            opt.enable_fp16 = true;
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
        case Backend::Trt: return "trt";
        case Backend::MnnCpu: return "mnn_cpu";
        case Backend::MnnCuda: return "mnn_cuda";
        case Backend::MnnOpencl: return "mnn_opencl";
        case Backend::MnnVulkan: return "mnn_vulkan";
        case Backend::SophgoF16: return "sophgo_tpu_f16";
        case Backend::SophgoInt8: return "sophgo_tpu_int8";
    }
    return "?";
}

bool is_sophgo(Backend b) { return b == Backend::SophgoF16 || b == Backend::SophgoInt8; }

// 后端感知模型路径：ORT 用 .onnx, MNN 用 .mnn, 原生 TRT 用 .engine, Sophgo 用 .bmodel。
// mnn/engine 传空串表示该模型未转换，对应后端运行时加载 onnx（会失败，便于外部对照补充）。
std::string model_path(const std::string& onnx, const std::string& mnn,
                       const std::string& engine, const std::string& bmodel, Backend b) {
    if (is_sophgo(b)) return "../../test_data/test_models/sophgo/" + bmodel;
    if (b == Backend::Trt)
        return engine.empty() ? "../../test_data/test_models/onnx/" + onnx
                              : "../../test_data/test_models/trt/" + engine;
    if (b == Backend::MnnCpu || b == Backend::MnnCuda ||
        b == Backend::MnnOpencl || b == Backend::MnnVulkan)
        return mnn.empty() ? "../../test_data/test_models/onnx/" + onnx
                           : "../../test_data/test_models/mnn/" + mnn;
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
    std::string model = model_path("yolo26n/yolo26n.onnx", "yolo11n_nms.mnn",
                                   "yolo26n.engine", "zhgd_without_nms_640_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b, "x:1x3x640x640");
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
    bench(*det, im, &res, 20, 100);
    modeldeploy::vision::dis_det(res);
    auto vis = modeldeploy::vision::vis_det(im, res, 0.5, label_map, kFont, 12, 0.3, false);
    (void)vis.imwrite("result_detection_" + std::string(backend_tag(b)) + ".jpg");
    std::printf("[%s] done, %zu objects\n", backend_tag(b), res.size());
    return 0;
}

int run_classification(Backend b) {
    std::string model = model_path("zhgd_ml.onnx", "yolo11n-cls.mnn",
                                   "yolo11n-cls.engine", "zhgd_ml_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b, "x:1x3x192x256");
    auto m = std::make_unique<modeldeploy::vision::classification::Classification>(model, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    const bool is_yolo_cls = (b == Backend::MnnCpu || b == Backend::MnnCuda ||
                              b == Backend::MnnOpencl || b == Backend::MnnVulkan ||
                              b == Backend::Trt);
    m->get_preprocessor().set_size(is_yolo_cls ? std::vector<int>{224, 224}
                                               : std::vector<int>{192, 256});
    m->get_preprocessor().disable_center_crop();
    // 后处理自动判别单/多标签：yolo11n-cls(单标签, softmax 和≈1)与 zhgd_ml(多标签, 和≈2)。
    m->get_postprocessor().set_multi_label_auto(true);
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
    std::string model = model_path("yolo11n/yolo11n-pose.onnx", "yolo11n-pose.mnn",
                                   "yolo11n-pose.engine", "yolo26n/yolo26n-pose_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b, "x:1x3x640x640");
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
    std::string model = model_path("yolo11n/yolo11n-obb_nms.onnx", "yolo11n-obb_nms.mnn",
                                   "yolo11n-obb_nms.engine", "yolo26n/yolo26n-obb_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b, "x:1x3x640x640");
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
    std::string model = model_path("yolo11n/yolo11n-seg_nms.onnx", "yolo11n-seg_nms.mnn",
                                   "yolo11n-seg_nms.engine", "yolo26n/yolo26n-seg_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b, "x:1x3x640x640");
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
    // sem 暂无 .mnn/.engine，仅 ORT/Sophgo 可用
    std::string model = model_path("yolo26n/yolo26n-sem.onnx", "", "",
                                   "yolo26n/yolo26n-sem_int8.bmodel", b);
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
    // depth 暂无 .mnn/.engine，仅 ORT/Sophgo 可用
    std::string model = model_path("yolo26n/yolo26n-depth.onnx", "", "",
                                   "yolo26n/yolo26n-depth_int8.bmodel", b);
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
    // face_det 暂无 .mnn，TRT 有预构建 scrfd_2.5g.engine
    std::string model = model_path("face/scrfd_2.5g_bnkps_shape640x640.onnx", "",
                                   "scrfd_2.5g.engine", "face/scrfd_2.5g_int8.bmodel", b);
    modeldeploy::RuntimeOption opt = make_option(b, "x:1x3x640x640");
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
    // lpr 暂无 .mnn；TRT 有预构建 yolov5plate/plate_recognition_color.engine
    std::string det = model_path("yolov5plate.onnx", "", "yolov5plate.engine", "", b);
    std::string rec = model_path("plate_recognition_color.onnx", "", "plate_recognition_color.engine", "", b);
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
    const char* dict = b == Backend::Trt  ? "../../test_data/ppocrv4_dict.txt"
                     : (b == Backend::MnnCpu || b == Backend::MnnCuda ||
                        b == Backend::MnnOpencl || b == Backend::MnnVulkan)
                         ? "../../test_data/ppocrv5_dict.txt"
                         : "../../test_data/dict.txt";
    // ORT: onnx；MNN: test_data/test_models/onnx/ocr/ppocrv5_mobile/*.mnn（OCR 的 .mnn 位于 onnx 目录下）
    // TRT: test_data/test_models/trt/ocr_*.engine
    std::string det, cls, rec;
    if (b == Backend::MnnCpu || b == Backend::MnnCuda ||
        b == Backend::MnnOpencl || b == Backend::MnnVulkan) {
        det = "../../test_data/test_models/onnx/ocr/ppocrv5_mobile/det_infer.mnn";
        cls = "../../test_data/test_models/onnx/ocr/ppocrv5_mobile/cls_infer.mnn";
        rec = "../../test_data/test_models/onnx/ocr/ppocrv5_mobile/rec_infer.mnn";
    } else if (b == Backend::Trt) {
        det = "../../test_data/test_models/trt/ocr_det.engine";
        cls = "../../test_data/test_models/trt/ocr_cls.engine";
        rec = "../../test_data/test_models/trt/ocr_rec.engine";
    } else if (is_sophgo(b)) {
        det = "../../test_data/test_models/sophgo/ocr_det.bmodel";
        cls = "../../test_data/test_models/sophgo/ocr_cls.bmodel";
        rec = "../../test_data/test_models/sophgo/ocr_rec.bmodel";
    } else {
        det = "../../test_data/test_models/onnx/ocr/ppocrv5_mobile/det_infer2.onnx";
        cls = "../../test_data/test_models/onnx/ocr/ppocrv4_mobile/cls_infer.onnx";
        rec = "../../test_data/test_models/onnx/ocr/ppocrv5_mobile/rec_infer1.onnx";
    }
    modeldeploy::RuntimeOption opt = make_option(b);
    auto m = std::make_unique<modeldeploy::vision::ocr::PaddleOCR>(det, cls, rec, dict, opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "[%s] init failed\n", backend_tag(b)); return 1; }
    m->set_rec_batch_size(8);
    auto& dpp = m->get_detector()->get_preprocessor();
    dpp.set_max_side_len(b == Backend::Trt ? 1280 : 1440);
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
    // ped_attr 暂无 .mnn；TRT 有 zhgd_det/zhgd_ml.engine
    std::string det = model_path("zhgd_det.onnx", "", "zhgd_det.engine", "", b);
    std::string ml = model_path("zhgd_ml.onnx", "", "zhgd_ml.engine", "", b);
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
