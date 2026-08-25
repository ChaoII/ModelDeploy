//
// capi 人脸年龄示例
//
#include "../capi_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_FACE_AGE,
                        "../../test_data/test_models/onnx/face/age_predictor.onnx", opt),
        "create face age");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_face_id1.jpg"), "read image");

    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    int age = 0;
    die(md_result_age(res, &age), "get age");
    std::printf("[capi] age: %d\n", age);

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    return 0;
}
