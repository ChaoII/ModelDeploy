// ModelDeploy demo: 多目标跟踪（MOT，ort_gpu_cuda_ep）。
// 与 demo_tracking_ort_cpu 同逻辑，仅后端不同：ORT + CUDA Execution Provider（GPU）。
#include "demo_tracking_impl.h"

int main() {
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_gpu(0);
    return run_tracking_demo(opt,
                             "../../test_data/test_models/onnx/yolo26n/yolo26n.onnx",
                             "result_tracking_ort_gpu_cuda_ep.jpg",
                             "ort_gpu_cuda_ep");
}
