//
// capi2 人脸防伪 pipeline（scrfd + fas_first + fas_second）示例
//
// 注意：capi2 中 MD_MODEL_FACE_AS_PIPELINE 尚未实现（md_model_create 返回
// MD_ERR_UNSUPPORTED_TYPE），因此 capi2 目前无法输出 REAL/FUZZY/SPOOF 判定。
// 本示例如实报告该限制，不伪造防伪结果。
//
#include "../capi2_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    MDStatus st = md_model_create(&model, MD_MODEL_FACE_AS_PIPELINE,
                                  "../../test_data/test_models/onnx/face/scrfd_2.5g_bnkps_shape640x640.onnx|"
                                  "../../test_data/test_models/onnx/face/fas_first.onnx|"
                                  "../../test_data/test_models/onnx/face/fas_second.onnx", opt);
    if (st != MD_OK) {
        std::fprintf(stderr,
                     "[capi2] face anti-spoof pipeline unavailable in capi2: "
                     "MD_MODEL_FACE_AS_PIPELINE is not implemented (need upstream wiring to "
                     "SeetaFaceAsPipeline) -> %s\n", md_get_last_error());
        md_option_destroy(opt);
        return 0;
    }

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_face_detection4.jpg"), "read image");
    MDResultHandle res = nullptr;
    st = md_model_predict(model, img, &res);
    if (st != MD_OK)
        std::fprintf(stderr, "[capi2] predict failed: %s\n", md_get_last_error());
    else
        md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    return 0;
}
