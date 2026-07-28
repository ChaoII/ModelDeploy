//
// 逐阶段耗时分析
//
#include "csrc/vision.h"
#include <iostream>

using namespace modeldeploy::vision;

int main() {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(4);
    option.use_ort_backend();
    option.use_gpu(0);
    option.enable_fp16 = true;
    option.ort_option.trt_engine_cache_path = "./trt_engine";

    auto det = modeldeploy::vision::detection::UltralyticsDet(
        "../../test_data/test_models/yolo11n_nms.onnx", option);
    det.get_preprocessor().use_cuda_preproc();
    det.get_preprocessor().set_size({640, 640});

    auto img = ImageData::imread("../../test_data/test_images/test_detection0.jpg");

    std::vector<DetectionResult> result;
    for (int i = 0; i < 10; ++i) det.predict(img, &result);

    const int N = 100;
    TimerArray timers;
    for (int i = 0; i < N; ++i) {
        det.predict(img, &result, &timers);
    }
    timers.print_benchmark();

    return 0;
}
