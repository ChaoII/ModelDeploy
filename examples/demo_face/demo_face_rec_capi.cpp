//
// capi 人脸识别（embedding 提取）示例
//
#include "../capi_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_FACE_REC,
                        "../../test_data/test_models/onnx/face/face_recognizer.onnx", opt),
        "create face rec");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_face_id4.jpg"), "read image");

    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    size_t count = 0;
    die(md_result_count(res, &count), "get count");
    for (size_t i = 0; i < count; ++i) {
        const float* emb = nullptr;
        size_t n = 0;
        die(md_result_face_embedding(res, i, &emb, &n), "get embedding");
        std::printf("[capi] face[%zu] embedding dim=%zu\n", i, n);
        std::printf("  first values:");
        for (size_t k = 0; k < n && k < 8; ++k)
            std::printf(" %.4f", emb[k]);
        std::printf("\n");
    }

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    return 0;
}
