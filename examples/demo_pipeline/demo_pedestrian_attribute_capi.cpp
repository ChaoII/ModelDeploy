//
// capi2 行人属性示例：检测 + 属性分类
//
#include "../capi2_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_PED_ATTR,
                        "../../test_data/test_models/onnx/zhgd_det.onnx|"
                        "../../test_data/test_models/onnx/zhgd_ml.onnx", opt),
        "create pedestrian attribute");

    die(md_model_set_param_d(model, "det_threshold", 0.5), "set det_threshold");
    die(md_model_set_cls_input_size(model, 192, 256), "set cls input size");
    die(md_model_set_input_size(model, 1280, 1280), "set det input size");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_pedestrian_attribute1.jpg"),
        "read image");

    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    const MDAttrItem* items = nullptr;
    size_t n = 0;
    die(md_result_attribute(res, &items, &n), "get attribute result");
    std::printf("detected %zu persons\n", n);
    for (size_t i = 0; i < n; ++i) {
        const float* scores = nullptr;
        size_t sn = 0;
        die(md_result_attr_scores(res, i, &scores, &sn), "get attr scores");
        std::printf("  [%zu] box=%.0f,%.0f,%.0f,%.0f box_score=%.3f attrs=(",
                    i, items[i].x, items[i].y, items[i].w, items[i].h, items[i].box_score);
        for (size_t k = 0; k < sn; ++k)
            std::printf("%s%.3f", k ? "," : "", scores[k]);
        std::printf(")\n");
    }

    MDDrawOptions draw{};
    draw.threshold = 0.5;
    draw.font_path = "../../test_data/msyh.ttc";
    draw.save_result = 1;
    die(md_draw_result(img, res, &draw), "draw");
    die(md_image_save(img, "capi2_attr_out.jpg"), "save");

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    std::puts("OK -> capi2_attr_out.jpg");
    return 0;
}
