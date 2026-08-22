// ModelDeploy demo: 多目标跟踪（MOT，mnn_cpu）。
// 与 demo_tracking_ort_cpu 同逻辑，仅后端不同：MNN CPU。
#include "demo_tracking_impl.h"

int main() {
    modeldeploy::RuntimeOption opt;
    opt.use_mnn_backend();
    opt.use_cpu();
    return run_tracking_demo(opt,
                             "../../test_data/test_models/mnn/yolo26n/yolo26n.mnn",
                             "result_tracking_mnn_cpu.jpg",
                             "mnn_cpu");
}
