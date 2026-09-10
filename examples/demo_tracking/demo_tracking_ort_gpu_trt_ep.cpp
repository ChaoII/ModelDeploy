// ModelDeploy demo: 多目标跟踪（MOT，ort_gpu_trt_ep）。
// 与 demo_tracking_ort_cpu 同逻辑，仅后端不同：ORT + TensorRT Execution Provider（GPU，在线建 engine）。
#include "demo_tracking_impl.h"

int main() {
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.set_device(modeldeploy::Device::GPU, 0);
    opt.enable_trt = true;
    opt.enable_fp16 = true;
    opt.ort_option.trt_engine_cache_path = "./trt_engine";
    return run_tracking_demo(opt,
                             "../../test_data/test_models/onnx/yolo26n/yolo26n.onnx",
                             "result_tracking_ort_gpu_trt_ep.jpg",
                             "ort_gpu_trt_ep");
}
