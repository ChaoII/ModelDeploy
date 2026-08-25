//
// capi 图像分类示例：演示 md_model_set_param_i/b
//
#include "../capi_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_CLASSIFICATION,
                        "../../test_data/test_models/onnx/yolo11n/yolo11n-cls.onnx", opt), "create classification");

    // 演示新参数 API
    die(md_model_set_param_i(model, "top_k", 5), "set top_k");
    die(md_model_set_param_b(model, "multi_label", 0), "set multi_label");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/bus.jpg"), "read image");

    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    const MDClassifyItem* items = nullptr;
    size_t n = 0;
    die(md_result_classification(res, &items, &n), "get classification result");
    std::printf("top %zu classes\n", n);
    for (size_t i = 0; i < n; ++i)
        std::printf("  [%d] score=%.3f\n", items[i].label_id, items[i].score);

    MDDrawOptions draw{};
    draw.threshold = 0.4;
    draw.font_path = "../../test_data/msyh.ttc";
    draw.save_result = 1;
    die(md_draw_result(img, res, &draw), "draw");
    die(md_image_save(img, "capi_cls_out.jpg"), "save");

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    std::puts("OK -> capi_cls_out.jpg");
    return 0;
}
