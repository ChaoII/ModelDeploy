//
// capi2 实例分割示例：演示 md_model_set_param_d
//
#include "../capi2_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_INSTANCE_SEG,
                        "../../test_data/test_models/onnx/yolo11n/yolo11n-seg.onnx", opt), "create instance seg");

    // 演示新参数 API
    die(md_model_set_param_d(model, "conf_threshold", 0.4), "set conf_threshold");
    die(md_model_set_param_d(model, "nms_threshold", 0.45), "set nms_threshold");
    die(md_model_set_param_d(model, "mask_threshold", 0.5), "set mask_threshold");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_detection0.jpg"), "read image");

    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    const MDIsegItem* items = nullptr;
    size_t n = 0;
    die(md_result_instance_seg(res, &items, &n), "get iseg result");
    std::printf("detected %zu instances\n", n);
    for (size_t i = 0; i < n; ++i) {
        std::printf("  [%d] score=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
                    items[i].label_id, items[i].score,
                    items[i].x, items[i].y, items[i].w, items[i].h);
        const unsigned char* mask = nullptr;
        size_t h = 0, w = 0;
        die(md_result_mask(res, i, &mask, &h, &w), "get mask");
        std::printf("    mask %zux%zu\n", h, w);
    }

    MDDrawOptions draw{};
    draw.threshold = 0.4;
    draw.font_path = "../../test_data/msyh.ttc";
    draw.save_result = 1;
    die(md_draw_result(img, res, &draw), "draw");
    die(md_image_save(img, "capi2_iseg_out.jpg"), "save");

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    std::puts("OK -> capi2_iseg_out.jpg");
    return 0;
}
