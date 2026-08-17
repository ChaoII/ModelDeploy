//
// capi2 人脸识别 pipeline（检测 + 特征提取）示例
//
#include "../capi2_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_FACE_REC_PIPELINE,
                        "../../test_data/test_models/onnx/face/scrfd_2.5g_bnkps_shape640x640.onnx|"
                        "../../test_data/test_models/onnx/face/face_recognizer.onnx", opt),
        "create face rec pipeline");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_face_detection4.jpg"), "read image");

    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    size_t count = 0;
    die(md_result_count(res, &count), "get count");
    std::printf("[capi2] detected %zu faces\n", count);
    for (size_t i = 0; i < count; ++i) {
        const float* emb = nullptr;
        size_t n = 0;
        die(md_result_face_embedding(res, i, &emb, &n), "get embedding");
        std::printf("  face[%zu] embedding dim=%zu\n", i, n);
    }

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    return 0;
}
