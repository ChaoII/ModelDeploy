//
// capi 姿态估计示例：演示 md_model_set_param_d/i
//
#include "../capi_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_POSE,
                        "../../test_data/test_models/onnx/yolo11n/yolo11n-pose.onnx", opt), "create pose");

    // 演示新参数 API
    die(md_model_set_param_d(model, "conf_threshold", 0.4), "set conf_threshold");
    die(md_model_set_param_i(model, "keypoints_num", 17), "set keypoints_num");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/bus.jpg"), "read image");

    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    const MDPoseItem* items = nullptr;
    size_t n = 0;
    die(md_result_pose(res, &items, &n), "get pose result");
    std::printf("detected %zu persons\n", n);
    for (size_t i = 0; i < n; ++i) {
        std::printf("  [%zu] score=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
                    i, items[i].score, items[i].x, items[i].y, items[i].w, items[i].h);
        const MDPoint3* kps = nullptr;
        size_t kn = 0;
        die(md_result_keypoints(res, i, &kps, &kn), "get keypoints");
        for (size_t k = 0; k < kn; ++k)
            std::printf("    kpt[%zu]=(%.0f,%.0f,%.2f)\n", k, kps[k].x, kps[k].y, kps[k].z);
    }

    MDDrawOptions draw{};
    draw.threshold = 0.4;
    draw.font_path = "../../test_data/msyh.ttc";
    draw.save_result = 1;
    die(md_draw_result(img, res, &draw), "draw");
    die(md_image_save(img, "capi_pose_out.jpg"), "save");

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    std::puts("OK -> capi_pose_out.jpg");
    return 0;
}
