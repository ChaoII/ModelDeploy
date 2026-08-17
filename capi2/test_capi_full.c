/* capi2 综合验证：覆盖主要模型 kind 的完整流程（数组式 getter） */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "md_capi.h"

static int fails = 0;

static void check(const char* name, MDStatus st) {
    if (st != MD_OK) {
        printf("  [FAIL] %s: %s\n", name, md_get_last_error());
        fails++;
    } else {
        printf("  [ok]   %s\n", name);
    }
}

static void run_det(const char* model_path, const char* img) {
    printf("== Detection ==\n");
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle m = NULL;
    check("create", md_model_create(&m, MD_MODEL_DETECTION, model_path, opt));
    if (m) {
        MDImageHandle img_h = NULL;
        check("image", md_image_from_file(&img_h, img));
        MDResultHandle res = NULL;
        check("predict", md_model_predict(m, img_h, &res));
        if (res) {
            const MDDetectionItem* items = NULL;
            size_t n = 0;
            check("getter", md_result_detection(res, &items, &n));
            printf("  count=%zu\n", n);
            for (size_t i = 0; i < n && i < 5; ++i) {
                printf("    [%zu] box=(%.0f,%.0f,%.0f,%.0f) label=%d score=%.3f\n",
                       i, items[i].x, items[i].y, items[i].w, items[i].h,
                       items[i].label_id, items[i].score);
            }
            md_result_destroy(res);
        }
        md_image_destroy(img_h);
        md_model_destroy(m);
    }
    md_option_destroy(opt);
}

static void run_cls(const char* model_path, const char* img) {
    printf("== Classification ==\n");
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle m = NULL;
    check("create", md_model_create(&m, MD_MODEL_CLASSIFICATION, model_path, opt));
    if (m) {
        MDImageHandle img_h = NULL;
        check("image", md_image_from_file(&img_h, img));
        MDResultHandle res = NULL;
        check("predict", md_model_predict(m, img_h, &res));
        if (res) {
            const MDClassifyItem* items = NULL;
            size_t n = 0;
            check("getter", md_result_classification(res, &items, &n));
            printf("  count=%zu\n", n);
            for (size_t i = 0; i < n && i < 5; ++i) {
                printf("    [%zu] label=%d score=%.3f\n", i, items[i].label_id, items[i].score);
            }
            md_result_destroy(res);
        }
        md_image_destroy(img_h);
        md_model_destroy(m);
    }
    md_option_destroy(opt);
}

static void run_pose(const char* model_path, const char* img) {
    printf("== Pose ==\n");
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle m = NULL;
    check("create", md_model_create(&m, MD_MODEL_POSE, model_path, opt));
    if (m) {
        MDImageHandle img_h = NULL;
        check("image", md_image_from_file(&img_h, img));
        MDResultHandle res = NULL;
        check("predict", md_model_predict(m, img_h, &res));
        if (res) {
            const MDPoseItem* items = NULL;
            size_t n = 0;
            check("getter", md_result_pose(res, &items, &n));
            printf("  count=%zu\n", n);
            for (size_t i = 0; i < n && i < 3; ++i) {
                const MDPoint3* kps = NULL;
                size_t kps_n = 0;
                md_result_keypoints(res, i, &kps, &kps_n);
                printf("    [%zu] box=(%.0f,%.0f,%.0f,%.0f) score=%.3f kps=%zu\n",
                       i, items[i].x, items[i].y, items[i].w, items[i].h, items[i].score, kps_n);
                if (kps_n > 0 && kps)
                    printf("      kp0=(%.1f,%.1f,%.1f)\n", kps[0].x, kps[0].y, kps[0].z);
            }
            md_result_destroy(res);
        }
        md_image_destroy(img_h);
        md_model_destroy(m);
    }
    md_option_destroy(opt);
}

static void run_obb(const char* model_path, const char* img) {
    printf("== OBB ==\n");
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle m = NULL;
    check("create", md_model_create(&m, MD_MODEL_OBB, model_path, opt));
    if (m) {
        MDImageHandle img_h = NULL;
        check("image", md_image_from_file(&img_h, img));
        MDResultHandle res = NULL;
        check("predict", md_model_predict(m, img_h, &res));
        if (res) {
            const MDObbItem* items = NULL;
            size_t n = 0;
            check("getter", md_result_obb(res, &items, &n));
            printf("  count=%zu\n", n);
            for (size_t i = 0; i < n && i < 3; ++i) {
                printf("    [%zu] rbox=(%.0f,%.0f,%.0f,%.0f,ang=%.1f) label=%d score=%.3f\n",
                       i, items[i].cx, items[i].cy, items[i].w, items[i].h, items[i].angle,
                       items[i].label_id, items[i].score);
            }
            md_result_destroy(res);
        }
        md_image_destroy(img_h);
        md_model_destroy(m);
    }
    md_option_destroy(opt);
}

static void run_seg(const char* model_path, const char* img) {
    printf("== InstanceSeg ==\n");
    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);

    MDModelHandle m = NULL;
    check("create", md_model_create(&m, MD_MODEL_INSTANCE_SEG, model_path, opt));
    if (m) {
        MDImageHandle img_h = NULL;
        check("image", md_image_from_file(&img_h, img));
        MDResultHandle res = NULL;
        check("predict", md_model_predict(m, img_h, &res));
        if (res) {
            const MDIsegItem* items = NULL;
            size_t n = 0;
            check("getter", md_result_instance_seg(res, &items, &n));
            printf("  count=%zu\n", n);
            for (size_t i = 0; i < n && i < 3; ++i) {
                const unsigned char* mask = NULL;
                size_t mh = 0, mw = 0;
                md_result_mask(res, i, &mask, &mh, &mw);
                printf("    [%zu] box=(%.0f,%.0f,%.0f,%.0f) label=%d score=%.3f mask=%zux%zu\n",
                       i, items[i].x, items[i].y, items[i].w, items[i].h,
                       items[i].label_id, items[i].score, mh, mw);
            }
            md_result_destroy(res);
        }
        md_image_destroy(img_h);
        md_model_destroy(m);
    }
    md_option_destroy(opt);
}

static void run_insightface(void) {
    printf("== InsightFace ==\n");
    const char* dir = "test_data/test_models/onnx/insightface/buffalo_l/";
    char path[1024];
    snprintf(path, sizeof(path), "%sdet_10g.onnx|%sw600k_r50.onnx|%s2d106det.onnx|%s1k3d68.onnx|%sgenderage.onnx",
             dir, dir, dir, dir, dir);

    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle m = NULL;
    check("create", md_model_create(&m, MD_MODEL_INSIGHTFACE, path, opt));
    if (m) {
        MDImageHandle img_h = NULL;
        check("image", md_image_from_file(&img_h, "test_data/test_images/test_face1.jpg"));
        MDResultHandle res = NULL;
        check("predict", md_model_predict(m, img_h, &res));
        if (res) {
            const MDInsightFaceItem* items = NULL;
            size_t n = 0;
            check("getter", md_result_insightface(res, &items, &n));
            printf("  count=%zu\n", n);
            for (size_t i = 0; i < n && i < 3; ++i) {
                const MDPoint* kps = NULL; size_t kps_n = 0;
                const float* emb = NULL; size_t emb_n = 0;
                const float* pose = NULL; size_t pose_n = 0;
                md_result_insightface_kps(res, i, &kps, &kps_n);
                md_result_insightface_embedding(res, i, &emb, &emb_n);
                md_result_insightface_pose(res, i, &pose, &pose_n);
                printf("    [%zu] box=(%.0f,%.0f,%.0f,%.0f) score=%.3f kps=%zu emb=%zu gender=%d age=%d pose=%zu\n",
                       i, items[i].x, items[i].y, items[i].w, items[i].h, items[i].score,
                       kps_n, emb_n, items[i].gender, items[i].age, pose_n);
            }
            md_result_destroy(res);
        }
        md_image_destroy(img_h);
        md_model_destroy(m);
    }
    md_option_destroy(opt);
}

static void run_ocr(const char* model_dir, const char* img) {
    printf("== OCR ==\n");
    char path[1024];
    snprintf(path, sizeof(path), "%sdet_infer.onnx|%scls_infer.onnx|%srec_infer.onnx|test_data/ppocrv4_dict.txt",
             model_dir, model_dir, model_dir);

    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle m = NULL;
    check("create", md_model_create(&m, MD_MODEL_OCR, path, opt));
    if (m) {
        MDImageHandle img_h = NULL;
        check("image", md_image_from_file(&img_h, img));
        MDResultHandle res = NULL;
        check("predict", md_model_predict(m, img_h, &res));
        if (res) {
            size_t n = 0;
            md_result_count(res, &n);
            printf("  count=%zu\n", n);
            for (size_t i = 0; i < n && i < 8; ++i) {
                const int* quad = NULL; const char* text = NULL; float score = 0;
                check("ocr getter", md_result_ocr(res, i, &quad, &text, &score));
                printf("    [%zu] '%s' score=%.3f quad=(%d,%d,%d,%d)\n",
                       i, text ? text : "", score, quad[0], quad[1], quad[2], quad[3]);
            }
            md_result_destroy(res);
        }
        md_image_destroy(img_h);
        md_model_destroy(m);
    }
    md_option_destroy(opt);
}

static void run_image_tools(void) {
    printf("== Image Tools ==\n");
    MDImageHandle img = NULL;
    check("from_file", md_image_from_file(&img, "test_data/test_images/test_detection0.jpg"));
    if (img) {
        int w = 0, h = 0;
        md_image_size(img, &w, &h);
        printf("  size=%dx%d\n", w, h);
        MDImageHandle clone = NULL;
        check("clone", md_image_clone(img, &clone));
        if (clone) {
            int cw = 0, ch = 0;
            md_image_size(clone, &cw, &ch);
            printf("  clone size=%dx%d\n", cw, ch);
            check("save", md_image_save(clone, "md_v2_test_out.png"));
            MDImageHandle crop = NULL;
            check("crop", md_image_crop(clone, 10, 10, 100, 80, &crop));
            if (crop) {
                int rw = 0, rh = 0;
                md_image_size(crop, &rw, &rh);
                printf("  crop size=%dx%d\n", rw, rh);
                const unsigned char* enc = NULL;
                size_t enc_n = 0;
                check("encode", md_image_encode(crop, ".jpg", &enc, &enc_n));
                printf("  encoded bytes=%zu\n", enc_n);
                md_image_destroy(crop);
            }
            md_image_destroy(clone);
        }
        md_image_destroy(img);
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    run_det("test_data/test_models/onnx/yolo11n/yolo11n.onnx",
            "test_data/test_images/test_detection0.jpg");
    run_cls("test_data/test_models/onnx/yolo11n/yolo11n-cls.onnx",
            "test_data/test_images/bus.jpg");
    run_pose("test_data/test_models/onnx/yolo11n/yolo11n-pose.onnx",
             "test_data/test_images/bus.jpg");
    run_obb("test_data/test_models/onnx/yolo11n/yolo11n-obb.onnx",
            "test_data/test_images/bus.jpg");
    run_seg("test_data/test_models/onnx/yolo11n/yolo11n-seg.onnx",
            "test_data/test_images/test_detection0.jpg");
    run_insightface();
    run_ocr("test_data/test_models/onnx/ocr/ppocrv4_mobile/", "test_data/test_images/ocr1.jpg");
    run_image_tools();

    printf("\n%s: %d failure(s)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
