#pragma once
namespace demo {
enum class Backend {
    OrtCpu, OrtGpuCudaEp, OrtGpuTrtEp,
    Trt,
    MnnCpu, MnnCuda, MnnOpencl, MnnVulkan,
    SophgoF16, SophgoInt8
};
int run_detection(Backend);
int run_classification(Backend);
int run_pose(Backend);
int run_obb(Backend);
int run_instance_seg(Backend);
int run_sem(Backend);
int run_depth(Backend);
int run_face_det(Backend);
int run_lpr_pipeline(Backend);
int run_ocr_pipeline(Backend);
int run_pedestrian_attribute(Backend);
}
