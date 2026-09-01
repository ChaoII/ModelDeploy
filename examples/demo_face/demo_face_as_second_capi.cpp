#include "../capi_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_FACE_AS_SECOND,
                        "../../test_data/test_models/onnx/seetaface/fas_second.onnx", opt),
        "create face anti-spoof second");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_face_as_second2.jpg"), "read image");
    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    size_t n = 0;
    die(md_result_count(res, &n), "count");
    for (size_t i = 0; i < n; ++i) {
        int label = 0;
        die(md_result_spoof(res, i, &label), "spoof");
        std::printf("face[%zu] anti-spoof: %s (label=%d)\n", i,
                    label == 0 ? "REAL" : (label == 2 ? "SPOOF" : "FUZZY"), label);
    }

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    return 0;
}
