// ModelDeploy demo: 多目标跟踪（MOT，trt）。
// 与 demo_tracking_ort_cpu 同逻辑，仅后端不同：原生 TensorRT（加载预构建 .engine）。
#include "demo_tracking_impl.h"

int main() {
    modeldeploy::RuntimeOption opt;
    opt.use_trt_backend();
    opt.set_device(modeldeploy::Device::GPU, 0);
    opt.enable_fp16 = true;
    return run_tracking_demo(opt,
                             "../../test_data/test_models/trt/yolo26n/yolo26n.engine",
                             "result_tracking_trt.jpg",
                             "trt");
}
