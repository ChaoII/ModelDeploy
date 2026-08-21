//
// capi 人脸检测示例：演示 md_model_set_param_d/i
//
#include "../capi_common.h"

int main() {
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_FACE_DET,
                        "../../test_data/test_models/onnx/face/scrfd_2.5g_bnkps_shape640x640.onnx", opt), "create face det");

    // 演示新参数 API
    die(md_model_set_param_d(model, "conf_threshold", 0.4), "set conf_threshold");
    die(md_model_set_param_d(model, "nms_threshold", 0.45), "set nms_threshold");
    die(md_model_set_param_i(model, "landmarks_per_face", 5), "set landmarks_per_face");

    MDImageHandle img = nullptr;
    die(md_image_from_file(&img, "../../test_data/test_images/test_face1.jpg"), "read image");

    MDResultHandle res = nullptr;
    die(md_model_predict(model, img, &res), "predict");

    const MDFaceItem* items = nullptr;
    size_t n = 0;
    die(md_result_face(res, &items, &n), "get face result");
    std::printf("detected %zu faces\n", n);
    for (size_t i = 0; i < n; ++i) {
        std::printf("  [%zu] score=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
                    i, items[i].score, items[i].x, items[i].y, items[i].w, items[i].h);
        const MDPoint* kps = nullptr;
        size_t kn = 0;
        die(md_result_face_kps(res, i, &kps, &kn), "get face kps");
        for (size_t k = 0; k < kn; ++k)
            std::printf("    kpt[%zu]=(%.0f,%.0f)\n", k, kps[k].x, kps[k].y);
    }

    MDDrawOptions draw{};
    draw.threshold = 0.4;
    draw.font_path = "../../test_data/msyh.ttc";
    draw.save_result = 1;
    MDStatus ds = md_draw_result(img, res, &draw);
    if (ds != MD_OK) {
        std::fprintf(stderr, "[capi] draw face_det unsupported (%s), saving original image\n", md_get_last_error());
    } else {
        die(md_image_save(img, "capi_face_det_out.jpg"), "save");
    }

    md_result_destroy(res);
    md_image_destroy(img);
    md_model_destroy(model);
    md_option_destroy(opt);
    if (ds == MD_OK)
        std::puts("OK -> capi_face_det_out.jpg");
    else
        std::puts("OK -> face detection printed (draw unsupported)");
    return 0;
}
