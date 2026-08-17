//
// capi2 LPR pipeline 示例：车牌检测 + 识别 + 颜色
//
#include "../capi2_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_LPR_PIPELINE,
                        "../../test_data/test_models/onnx/yolov5plate.onnx|"
                        "../../test_data/test_models/onnx/plate_recognition_color.onnx", opt),
        "create lpr pipeline");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_lpr_pipeline.jpg"), "read image");

    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    const MDLprItem* items = nullptr;
    size_t n = 0;
    die(md_result_lpr(res, &items, &n), "get lpr result");
    std::printf("detected %zu plates\n", n);
    for (size_t i = 0; i < n; ++i) {
        const char* plate = nullptr;
        const char* color = nullptr;
        die(md_result_plate(res, i, &plate, &color), "get plate");
        std::printf("  [%zu] score=%.3f plate=%s color=%s\n",
                    i, items[i].score, plate ? plate : "(null)", color ? color : "(null)");
    }

    MDDrawOptions draw{};
    draw.font_path = "../../test_data/msyh.ttc";
    draw.save_result = 1;
    die(md_draw_result(img, res, &draw), "draw");
    die(md_image_save(img, "capi2_lpr_out.jpg"), "save");

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    std::puts("OK -> capi2_lpr_out.jpg");
    return 0;
}
