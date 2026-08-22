// ModelDeploy demo: 多目标跟踪（MOT，ort_cpu）。
// 端到端流水线：构造 RuntimeOption -> 加载检测模型 -> 逐帧 predict -> 送入追踪器
// (ByteTracker) -> 可视化（画框 + 稳定的 track_id）-> 保存 jpg。
//
// 说明：无真实视频时，本 demo 用单张测试图模拟多帧序列——把同一批检测框按帧做轻微
// 确定性抖动作为"新一帧"，连续送入追踪器，从而演示同一物体在帧间保持稳定的 track_id。
#include "demo_tracking_impl.h"

int main() {
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    opt.set_cpu_thread_num(4);
    return run_tracking_demo(opt,
                             "../../test_data/test_models/onnx/yolo26n/yolo26n.onnx",
                             "result_tracking_ort_cpu.jpg",
                             "ort_cpu");
}
