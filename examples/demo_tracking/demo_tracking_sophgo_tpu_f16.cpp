// ModelDeploy demo: 多目标跟踪（MOT，sophgo_tpu_f16）。
// 与 demo_tracking_ort_cpu 同逻辑，仅后端不同：Sophgo TPU（f16 bmodel）。
#include "demo_tracking_impl.h"

int main() {
    modeldeploy::RuntimeOption opt;
    opt.use_sophgo_backend(0);
    return run_tracking_demo(opt,
                             "../../test_data/test_models/sophgo/yolo26n/yolo26n-f16.bmodel",
                             "result_tracking_sophgo_tpu_f16.jpg",
                             "sophgo_tpu_f16");
}
