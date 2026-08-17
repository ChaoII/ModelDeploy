//
// capi2 人脸防伪（first 级 / fas_first.onnx）示例
//
// 注意：capi2 中 MD_MODEL_FACE_AS 当前映射到 InsightFaceGenderAge（性别/年龄）
// 模型，而非防伪网络；MD_MODEL_FACE_AS_PIPELINE 尚未实现。因此 capi2 目前无法
// 输出真实的 REAL/SPOOF 判定。本示例如实创建 FACE_AS 模型并报告其状态，不伪造
// 防伪结果。
//
#include "../capi2_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    MDStatus st = md_model_create(&model, MD_MODEL_FACE_AS,
                                  "../../test_data/test_models/onnx/face/fas_first.onnx", opt);
    if (st != MD_OK) {
        std::fprintf(stderr,
                     "[capi2] face anti-spoof unavailable in capi2: "
                     "MD_MODEL_FACE_AS is wired to InsightFaceGenderAge (gender/age), "
                     "not the fas_first network -> %s\n", md_get_last_error());
        md_option_destroy(opt);
        return 0;
    }

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_face_id3.jpg"), "read image");
    MDResultHandle res = nullptr;
    st = md_model_predict(model, img, &res);
    if (st != MD_OK)
        std::fprintf(stderr, "[capi2] predict on genderage path failed: %s\n", md_get_last_error());
    else
        md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    return 0;
}
