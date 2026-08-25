//
// capi OCR 整链路示例：演示多子模型聚合创建与参数 API
//
#include "../capi_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_OCR,
                        "../../test_data/test_models/onnx/ocr/ppocrv4_mobile/det_infer.onnx|"
                        "../../test_data/test_models/onnx/ocr/ppocrv4_mobile/cls_infer.onnx|"
                        "../../test_data/test_models/onnx/ocr/ppocrv4_mobile/rec_infer.onnx|"
                        "../../test_data/ppocrv4_dict.txt", opt), "create ocr");

    die(md_model_set_param_d(model, "det_db_box_thresh", 0.6), "set box_thresh");
    die(md_model_set_param_d(model, "cls_thresh", 0.9), "set cls_thresh");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_ocr.png"), "read image");

    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    size_t n = 0;
    die(md_result_count(res, &n), "count");
    std::printf("detected %zu text lines\n", n);
    for (size_t i = 0; i < n; ++i) {
        const int* quad = nullptr;
        const char* text = nullptr;
        float score = 0.f;
        die(md_result_ocr(res, i, &quad, &text, &score), "get ocr result");
        if (text)
            std::printf("  [%zu] score=%.3f %s\n", i, score, text);
    }

    MDDrawOptions draw{};
    draw.font_path = "../../test_data/msyh.ttc";
    draw.save_result = 1;
    die(md_draw_result(img, res, &draw), "draw");
    die(md_image_save(img, "capi_ocr_out.jpg"), "save");

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    std::puts("OK -> capi_ocr_out.jpg");
    return 0;
}
