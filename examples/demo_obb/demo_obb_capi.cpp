//
// capi 旋转框（OBB）检测示例：演示 md_model_set_param_d
//
#include "../capi_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_OBB,
                        "../../test_data/test_models/onnx/yolo11n/yolo11n-obb.onnx", opt), "create obb");

    // 演示新参数 API
    die(md_model_set_param_d(model, "conf_threshold", 0.4), "set conf_threshold");
    die(md_model_set_param_d(model, "nms_threshold", 0.45), "set nms_threshold");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_obb1.jpg"), "read image");

    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    const MDObbItem* items = nullptr;
    size_t n = 0;
    die(md_result_obb(res, &items, &n), "get obb result");
    std::printf("detected %zu rotated boxes\n", n);
    for (size_t i = 0; i < n; ++i)
        std::printf("  [%d] score=%.3f center=(%.0f,%.0f) size=(%.0f,%.0f) angle=%.2f\n",
                    items[i].label_id, items[i].score,
                    items[i].cx, items[i].cy, items[i].w, items[i].h, items[i].angle);

    MDDrawOptions draw{};
    draw.threshold = 0.4;
    draw.font_path = "../../test_data/msyh.ttc";
    draw.save_result = 1;
    die(md_draw_result(img, res, &draw), "draw");
    die(md_image_save(img, "capi_obb_out.jpg"), "save");

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    std::puts("OK -> capi_obb_out.jpg");
    return 0;
}
