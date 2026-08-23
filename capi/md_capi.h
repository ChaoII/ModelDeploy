//
// ModelDeploy 纯 C API v2
//
// 设计原则：
//  1. 不透明句柄：调用方永远接触不到库内部指针/结构体字段，杜绝类型强转与字段篡改。
//  2. 单一分发点：模型创建/释放/推理各自只有一个入口，内部按类型分发，消灭重复样板。
//  3. 统一内存所有权：句柄一律由库分配/释放，结果统一 md_result_destroy，杜绝
//     malloc/new[]/strdup 混用导致的不配对 free。
//  4. 纯 C99：头文件不依赖 C++ 语法（无 bool/默认参数/引用），任何 C 编译器可编译。
//  5. 富错误模型：MDStatus 覆盖参数/状态/并发类错误，md_get_last_error() 提供线程安全
//     的错误信息。
//  6. 线程模型：每个句柄只能单线程使用；多线程并发需各自 create 句柄。库内部不加锁
//     以保持零开销。
//  7. 数组式结果访问：固定字段结果一次取回 blittable 结构体数组（零拷贝），可变长
//     数据（关键点/embedding/mask/字符串）按项取，每次仅一次调用。
//

#ifndef MD_CAPI_H
#define MD_CAPI_H

#include <stddef.h>
#include <stdint.h>

/* 导出宏（Windows dllexport / Linux visibility） */
#if defined(_WIN32)
#  if defined(MD_CAPI)
#    define MD_CAPI_EXPORT __declspec(dllexport)
#  else
#    define MD_CAPI_EXPORT __declspec(dllimport)
#  endif
#else
#  define MD_CAPI_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ==================== 错误模型 ==================== */

typedef enum MD_STATUS {
    MD_OK = 0,
    MD_ERR_NULL_POINTER,        /* 入参为空 */
    MD_ERR_INVALID_ARGUMENT,    /* 参数不合法（路径为空/尺寸非法等） */
    MD_ERR_PATH_NOT_FOUND,      /* 模型/文件不存在 */
    MD_ERR_MODEL_LOAD,          /* 模型加载失败（含后端初始化失败） */
    MD_ERR_MODEL_PREDICT,       /* 推理失败 */
    MD_ERR_MODEL_INIT,          /* 模型初始化失败（is_initialized()==false） */
    MD_ERR_UNSUPPORTED_TYPE,    /* 模型类型不支持 */
    MD_ERR_UNSUPPORTED_BACKEND, /* 后端不可用 */
    MD_ERR_OUT_OF_MEMORY,       /* 内存分配失败 */
    MD_ERR_IMAGE_DECODE,        /* 图像解码失败 */
    MD_ERR_BUSY,                /* 句柄并发使用（检测到非单线程） */
    MD_ERR_NOT_IMPLEMENTED,     /* 功能未实现 */
    MD_ERR_AUDIO_DECODE,        /* 音频解码失败 */
    MD_ERR_INVALID_TYPE         /* 参数类型不匹配 */
} MDStatus;

/* 线程安全地获取最近一次错误信息（thread_local，返回空串表示无错误） */
MD_CAPI_EXPORT const char* md_get_last_error(void);

/* ==================== 句柄类型（不透明：调用方只见指针，不可解引用） ==================== */

typedef struct md_model_handle* MDModelHandle;
typedef struct md_image_handle* MDImageHandle;
typedef struct md_result_handle* MDResultHandle;
typedef struct md_option_handle* MDOptionHandle;
typedef struct md_solution_handle* MDSolutionHandle;

/* ==================== 模型类型 ==================== */

typedef enum MD_MODEL_KIND {
    MD_MODEL_DETECTION = 0,
    MD_MODEL_CLASSIFICATION,
    MD_MODEL_POSE,
    MD_MODEL_OBB,
    MD_MODEL_INSTANCE_SEG,
    MD_MODEL_SEM_SEG,
    MD_MODEL_DEPTH,
    MD_MODEL_FACE_DET,
    MD_MODEL_FACE_REC,
    MD_MODEL_FACE_AGE,
    MD_MODEL_FACE_GENDER,
    MD_MODEL_FACE_AS,
    MD_MODEL_FACE_AS_PIPELINE,
    MD_MODEL_FACE_REC_PIPELINE,
    MD_MODEL_INSIGHTFACE,
    MD_MODEL_INSIGHTFACE_DET,
    MD_MODEL_OCR,
    MD_MODEL_OCR_DET,
    MD_MODEL_OCR_REC,
    MD_MODEL_OCR_CLS,
    MD_MODEL_LPR_DET,
    MD_MODEL_LPR_REC,
    MD_MODEL_LPR_PIPELINE,
    MD_MODEL_PED_ATTR,
    MD_MODEL_ASR,
    MD_MODEL_TTS,
    MD_MODEL_FACE_AS_SECOND,
    MD_MODEL_HAND,
    MD_MODEL_REID,
    MD_MODEL_SPEAKER_VERIFY,
    MD_MODEL_FORMULA_RECOGNIZER,
    MD_MODEL_TSN,
    MD_MODEL_ST_GCN,
    MD_MODEL_VEHICLE_KEYPOINT,
    MD_MODEL_FACE_LANDMARK,
    MD_MODEL_COUNT
} MDModelKind;

/* ==================== 运行时选项 ==================== */

/* 设备与后端 */
typedef enum MD_DEVICE {
    MD_DEV_CPU = 0,
    MD_DEV_GPU = 1,
    MD_DEV_TPU = 2,
    MD_DEV_OPENCL = 3,  /* 预留，未实现：md_option_set_device 会报错而非静默忽略 */
    MD_DEV_VULKAN = 4   /* 预留，未实现：md_option_set_device 会报错而非静默忽略 */
} MDDevice;

typedef enum MD_BACKEND {
    MD_BK_ORT = 0,
    MD_BK_MNN = 1,
    MD_BK_TRT = 2,
    MD_BK_SOPHGO = 3
} MDBackend;

/* 创建默认选项（可后续用 setter 覆盖） */
MD_CAPI_EXPORT MDStatus md_option_create(MDOptionHandle* out);
MD_CAPI_EXPORT void md_option_destroy(MDOptionHandle);

MD_CAPI_EXPORT void md_option_set_device(MDOptionHandle, MDDevice);
MD_CAPI_EXPORT void md_option_set_device_id(MDOptionHandle, int device_id);
MD_CAPI_EXPORT void md_option_set_backend(MDOptionHandle, MDBackend);
MD_CAPI_EXPORT void md_option_set_cpu_threads(MDOptionHandle, int n);
MD_CAPI_EXPORT void md_option_set_fp16(MDOptionHandle, int enable);
MD_CAPI_EXPORT void md_option_set_trt_engine_path(MDOptionHandle, const char* path);

/* ==================== 图像 ==================== */

/* 从文件读图（库内解码 + 分配，调用方只需 destroy） */
MD_CAPI_EXPORT MDStatus md_image_from_file(MDImageHandle* out, const char* path);

/* 从内存构造（data 由调用方持有，库只引用不拷贝；调用方保证生命周期） */
MD_CAPI_EXPORT MDStatus md_image_from_bgr24(MDImageHandle* out, const void* bgr, int w, int h);
MD_CAPI_EXPORT MDStatus md_image_from_rgb24(MDImageHandle* out, const void* rgb, int w, int h);
MD_CAPI_EXPORT MDStatus md_image_from_nv12(MDImageHandle* out, const void* y, const void* uv,
                            int w, int h, int step_y, int step_uv);
/* owned 版：拷入自有缓冲，产真 NV12 两平面帧（调用方无需保活） */
MD_CAPI_EXPORT MDStatus md_image_from_nv12_owned(MDImageHandle* out, const void* y, const void* uv,
                                  int w, int h, int step_y, int step_uv);
/* 从设备 NV12 两平面构造自描述 ImageData（零拷贝借用外部 y/uv，库不拥有内存）。
 * dev 指明帧所在设备（CPU/GPU/TPU）；step_y/step_uv<=0 时依 w 兜底。 */
MD_CAPI_EXPORT MDStatus md_image_from_device_nv12(MDImageHandle* out, const void* y, const void* uv,
                            int w, int h, int step_y, int step_uv, MDDevice dev);
MD_CAPI_EXPORT MDStatus md_image_from_yuv420p(MDImageHandle* out, const void* data, int w, int h);

/* 编码数据 / base64 / 压缩字节 */
MD_CAPI_EXPORT MDStatus md_image_from_encoded(MDImageHandle* out, const void* bytes, size_t n);
MD_CAPI_EXPORT MDStatus md_image_from_base64(MDImageHandle* out, const char* b64);

/* 深拷贝 / 裁剪 / 显示 / 保存 / 编码（输出 buffer 归图像句柄所有，随 destroy 释放） */
MD_CAPI_EXPORT MDStatus md_image_clone(MDImageHandle in, MDImageHandle* out);
MD_CAPI_EXPORT MDStatus md_image_crop(MDImageHandle in, int x, int y, int w, int h, MDImageHandle* out);
MD_CAPI_EXPORT MDStatus md_image_show(MDImageHandle);
MD_CAPI_EXPORT MDStatus md_image_save(MDImageHandle, const char* path);
MD_CAPI_EXPORT MDStatus md_image_encode(MDImageHandle, const char* ext,
                         const unsigned char** buf, size_t* n);

MD_CAPI_EXPORT void md_image_destroy(MDImageHandle);

/* 图像尺寸查询 */
MD_CAPI_EXPORT MDStatus md_image_size(MDImageHandle, int* w, int* h);

/* 图像元数据：type=MdImageType 数值, dev=MDDevice, nplanes=平面数（NV12=2, packed=1）；任一指针可为空 */
MD_CAPI_EXPORT MDStatus md_image_info(MDImageHandle, int* type, int* dev, int* nplanes);

/*
 * 取帧平面指针（供外部零拷贝读取/写入，典型于 NV12 设备帧）。
 *  - 仅对 NV12 类型有效；CPU BGR 图此处返回 MD_ERR_UNSUPPORTED_TYPE 且 y=uv=NULL。
 *  - dev 返回帧所在设备（CPU/GPU/TPU）。
 */
MD_CAPI_EXPORT MDStatus md_image_plane_ptrs(MDImageHandle, MDDevice* dev, void** y, void** uv);

/* ==================== 模型 ==================== */

/*
 * 统一创建模型。
 *  - kind: 模型类型（见 MD_MODEL_*）
 *  - model_path: 模型文件路径（onnx/mnn/engine/bmodel，库按扩展名推断后端，
 *    也可通过 option 强制指定后端）
 *  - 多子模型（OCR/LPR pipeline/insightface/audio）：model_path 用 '|' 分隔符串联：
 *      OCR:        det.onnx|cls.onnx|rec.onnx|dict.txt
 *      LPR pipeline: det.onnx|rec.onnx
 *      insightface:  det.onnx|rec.onnx|lmk2d.onnx|lmk3d.onnx[|genderage.onnx]
 *      ASR:        model.onnx|tokens.txt
 *      TTS:        model.onnx|tokens.txt|lex_en.txt|lex_zh.txt|voices.bin|jieba_dir|norm_dir
 */
MD_CAPI_EXPORT MDStatus md_model_create(MDModelHandle* out, MDModelKind kind,
                         const char* model_path, const MDOptionHandle opt);

MD_CAPI_EXPORT void md_model_destroy(MDModelHandle);

/* 深拷贝模型句柄（独立实例，可并行/独立使用；基于模型内部 clone()，组合模型重新加载） */
MD_CAPI_EXPORT MDStatus md_model_clone(MDModelHandle in, MDModelHandle* out);

/* 模型是否就绪 */
MD_CAPI_EXPORT MDStatus md_model_ready(MDModelHandle);

/* 设置模型输入尺寸（可选；默认按模型内置输入。pipeline 模型设置检测子模型尺寸） */
MD_CAPI_EXPORT MDStatus md_model_set_input_size(MDModelHandle, int w, int h);

/* 设置 pipeline 模型的分类子模型输入尺寸（当前仅 PedestrianAttribute 使用） */
MD_CAPI_EXPORT MDStatus md_model_set_cls_input_size(MDModelHandle, int w, int h);

/* 设置 pipeline 模型分类子模型的 batch 大小（>0 固定，-1 自动；PedestrianAttribute / OCR）。
 * 0 或 < -1 → MD_ERR_INVALID_ARGUMENT；非 pipeline kind → MD_ERR_UNSUPPORTED_TYPE */
MD_CAPI_EXPORT MDStatus md_model_set_cls_batch_size(MDModelHandle, int batch);

/* 设置 OCR 整链路识别子模型（rec）的 batch 大小（>0 固定，-1 自动）。
 * 0 或 < -1 → MD_ERR_INVALID_ARGUMENT；非 OCR kind → MD_ERR_UNSUPPORTED_TYPE */
MD_CAPI_EXPORT MDStatus md_model_set_rec_batch_size(MDModelHandle, int batch);

/* 设置 OCR 识别子模型输入形状 (c, h, w)（对应 rec_preprocessor::set_rec_image_shape）。
 * 仅 OCR_REC / OCR；其余 → MD_ERR_UNSUPPORTED_TYPE */
MD_CAPI_EXPORT MDStatus md_model_set_rec_image_shape(MDModelHandle, int c, int h, int w);

/* ==================== 模型前/后处理参数 ==================== */
/* 按扁平参数名设置模型前/后处理参数（模型级持久；未设置用默认值）。
 * _i 整型、_d 浮点、_b 布尔(0/1)、_s 字符串/枚举。不支持的 kind → MD_ERR_UNSUPPORTED_TYPE；
 * 未知名 → MD_ERR_INVALID_ARGUMENT；类型不匹配 → MD_ERR_INVALID_TYPE；s 且 value=null → MD_ERR_NULL_POINTER。 */
MD_CAPI_EXPORT MDStatus md_model_set_param_i(MDModelHandle, const char* name, int64_t value);
MD_CAPI_EXPORT MDStatus md_model_set_param_d(MDModelHandle, const char* name, double value);
MD_CAPI_EXPORT MDStatus md_model_set_param_b(MDModelHandle, const char* name, int enable);
MD_CAPI_EXPORT MDStatus md_model_set_param_s(MDModelHandle, const char* name, const char* value);

/* 自省：names 以 '|' 分隔的单个字符串（库持有，无需释放；kind 无参数→空串）；
 * type_out 输出 'I'/'D'/'B'/'S'；未知名 → MD_ERR_INVALID_ARGUMENT；kind 越界 → MD_ERR_INVALID_ARGUMENT。 */
MD_CAPI_EXPORT MDStatus md_model_param_names(MDModelKind kind, const char** names);
MD_CAPI_EXPORT MDStatus md_model_param_type(MDModelKind kind, const char* name, char* type_out);

/* 推理：统一入口（视觉），结果句柄由库分配，调用方用 md_result_destroy 释放 */
MD_CAPI_EXPORT MDStatus md_model_predict(MDModelHandle, MDImageHandle, MDResultHandle* out);

/* 批量推理（多图，仅支持的模型） */
MD_CAPI_EXPORT MDStatus md_model_predict_batch(MDModelHandle, MDImageHandle* imgs, size_t n,
                                MDResultHandle* out);

/* 动作识别：TSN 帧序列（多图）推理，结果 kind=MD_RES_CLASSIFICATION（复用 md_result_classification） */
MD_CAPI_EXPORT MDStatus md_model_predict_sequence(MDModelHandle h, MDImageHandle* frames, size_t n,
                                  MDResultHandle* out);
/* 动作识别：ST-GCN 骨骼序列推理（joints 为 T*V*C 行主序），结果 kind=MD_RES_CLASSIFICATION */
MD_CAPI_EXPORT MDStatus md_model_predict_skeleton(MDModelHandle h, const float* joints,
                                  size_t T, size_t V, size_t C, MDResultHandle* out);

/* ==================== 音频（ASR / TTS） ==================== */

/* ASR：从 wav 文件识别文本（库内解码 wav），text 归结果句柄所有无需释放 */
MD_CAPI_EXPORT MDStatus md_audio_asr_wav(MDModelHandle, const char* wav_path, const char** text);

/* ASR：从 PCM 浮点采样识别文本（data 由调用方持有，库只读） */
MD_CAPI_EXPORT MDStatus md_audio_asr(MDModelHandle, const float* samples, size_t n, int sample_rate,
                      const char** text);

/* TTS：文本合成音频（零拷贝返回库内 buffer；audio 归结果内部，随 md_model_destroy 释放前有效） */
MD_CAPI_EXPORT MDStatus md_audio_tts(MDModelHandle, const char* text, const char* voice, float speed,
                      int* sample_rate, const float** audio, size_t* audio_n);

/* 声纹（SpeakerVerify）：提取说话人 embedding（借用指针；embedding 归模型句柄内部，
 * 随 md_model_destroy 释放 / 下次 predict 覆盖前有效，与 TTS/ASR 暂存区生命周期一致） */
MD_CAPI_EXPORT MDStatus md_audio_speaker_embed(MDModelHandle, const float* samples, size_t n,
                              const float** embedding, size_t* emb_n);

/* wav 落盘辅助 */
MD_CAPI_EXPORT MDStatus md_wav_save(const float* samples, size_t n, int sample_rate, const char* path);

/* ==================== 结果（统一释放） ==================== */

MD_CAPI_EXPORT void md_result_destroy(MDResultHandle);

/* 结果种类（用于区分 getter） */
typedef enum MD_RESULT_KIND {
    MD_RES_DETECTION = 0,
    MD_RES_CLASSIFICATION,
    MD_RES_POSE,
    MD_RES_OBB,
    MD_RES_INSTANCE_SEG,
    MD_RES_SEM_SEG,
    MD_RES_DEPTH,
    MD_RES_FACE,
    MD_RES_FACE_REC,
    MD_RES_INSIGHTFACE,
    MD_RES_OCR,
    MD_RES_LPR,
    MD_RES_ATTR,
    MD_RES_AGE,
    MD_RES_GENDER,
    MD_RES_ASR,
    MD_RES_TTS,
    MD_RES_ANTISPOOF,
    MD_RES_REID,
    MD_RES_FORMULA
} MDResultKind;
MD_CAPI_EXPORT MDStatus md_result_kind(MDResultHandle, MDResultKind* out);
/* 结果对应的图内实例数（人脸数/文本行数/目标数等；单值结果恒为 1） */
MD_CAPI_EXPORT MDStatus md_result_count(MDResultHandle, size_t* out);

/* ==================== 通用几何 / 颜色 ==================== */

typedef struct MDBox { float x, y, w, h; } MDBox;
typedef struct MDPoint { float x, y; } MDPoint;
typedef struct MDPoint3 { float x, y, z; } MDPoint3;
typedef struct MDRotatedBox { float cx, cy, w, h, angle; } MDRotatedBox;
typedef struct MDRectF { float x, y, w, h; } MDRectF;
typedef struct MDColorRGBA { unsigned char r, g, b, a; } MDColorRGBA;

/* ==================== 结果项结构（blittable：纯数值、无指针，C# 可直映） ==================== */

typedef struct MDDetectionItem {
    float x, y, w, h;
    float score;
    int label_id;
} MDDetectionItem;

typedef struct MDClassifyItem {
    int label_id;
    float score;
} MDClassifyItem;

typedef struct MDPoseItem {
    float x, y, w, h;
    float score;
} MDPoseItem;

typedef struct MDObbItem {
    float cx, cy, w, h, angle;
    float score;
    int label_id;
} MDObbItem;

typedef struct MDIsegItem {
    float x, y, w, h;
    float score;
    int label_id;
} MDIsegItem;

typedef struct MDLprItem {
    float x, y, w, h;
    float score;
} MDLprItem;

typedef struct MDAttrItem {
    float x, y, w, h;
    float box_score;
    int box_label_id;
} MDAttrItem;

typedef struct MDFaceItem {
    float x, y, w, h;
    float score;
} MDFaceItem;

typedef struct MDInsightFaceItem {
    float x, y, w, h;
    float score;
    int gender;
    int age;
} MDInsightFaceItem;

/* ==================== 结果 getter（数组式：一次取回，内存归结果句柄所有） ==================== */

/* Detection */
MD_CAPI_EXPORT MDStatus md_result_detection(MDResultHandle, const MDDetectionItem** items, size_t* count);

/* Classification */
MD_CAPI_EXPORT MDStatus md_result_classification(MDResultHandle, const MDClassifyItem** items, size_t* count);

/* Pose（bbox + score 在数组项；骨架关键点在 md_result_keypoints） */
MD_CAPI_EXPORT MDStatus md_result_pose(MDResultHandle, const MDPoseItem** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_keypoints(MDResultHandle, size_t i, const MDPoint3** kps, size_t* n);

/* OBB */
MD_CAPI_EXPORT MDStatus md_result_obb(MDResultHandle, const MDObbItem** items, size_t* count);

/* InstanceSeg（mask 单列） */
MD_CAPI_EXPORT MDStatus md_result_instance_seg(MDResultHandle, const MDIsegItem** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_mask(MDResultHandle, size_t i, const unsigned char** buf, size_t* out_h, size_t* out_w);

/* SemSeg / Depth（整图单值；批量句柄下读 index 0） */
MD_CAPI_EXPORT MDStatus md_result_sem_seg(MDResultHandle, const unsigned char** labels, size_t* h, size_t* w,
                           int* num_classes);
MD_CAPI_EXPORT MDStatus md_result_depth(MDResultHandle, const float** depth, size_t* h, size_t* w);
/* 批量：按图索引取每图标签/深度 */
MD_CAPI_EXPORT MDStatus md_result_sem_seg_batch(MDResultHandle, size_t i, const unsigned char** labels, size_t* h, size_t* w,
                                 int* num_classes);
MD_CAPI_EXPORT MDStatus md_result_depth_batch(MDResultHandle, size_t i, const float** depth, size_t* h, size_t* w);

/* FaceDet（bbox + score + 关键点） */
MD_CAPI_EXPORT MDStatus md_result_face(MDResultHandle, const MDFaceItem** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_face_kps(MDResultHandle, size_t i, const MDPoint** kps, size_t* n);

/* FaceRec（embedding 单列，i 为实例序号） */
MD_CAPI_EXPORT MDStatus md_result_face_embedding(MDResultHandle, size_t i,
                                  const float** embedding, size_t* emb_n);

/* ReID（行人重识别 512-d embedding 单列，i 为实例序号） */
MD_CAPI_EXPORT MDStatus md_result_reid_embedding(MDResultHandle, size_t i,
                                  const float** embedding, size_t* emb_n);

/* InsightFace 完整分析 */
MD_CAPI_EXPORT MDStatus md_result_insightface(MDResultHandle, const MDInsightFaceItem** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_insightface_kps(MDResultHandle, size_t i, const MDPoint** kps, size_t* n);
MD_CAPI_EXPORT MDStatus md_result_insightface_embedding(MDResultHandle, size_t i,
                                         const float** embedding, size_t* emb_n);
MD_CAPI_EXPORT MDStatus md_result_insightface_pose(MDResultHandle, size_t i,
                                    const float** pose, size_t* n);

/* OCR（字符串按行取） */
MD_CAPI_EXPORT MDStatus md_result_ocr(MDResultHandle, size_t i, const int** quad, const char** text, float* score);

/* OCR 批量：返回 batch 内图像数（ResultData<OCRResult> size），逐图行读取复用 md_result_ocr */
MD_CAPI_EXPORT MDStatus md_result_ocr_batch_count(MDResultHandle, size_t* n);

/* OCR 方向分类（cls 子模型结果：label + score） */
MD_CAPI_EXPORT MDStatus md_result_ocr_cls(MDResultHandle, size_t i, int* cls_label, float* cls_score);

/* FormulaRecognizer（单块 LaTeX 文本；借用指针归结果句柄所有，随 md_result_destroy 失效） */
MD_CAPI_EXPORT MDStatus md_result_formula(MDResultHandle, size_t i, const char** latex);

/* LPR（plate 单列） */
MD_CAPI_EXPORT MDStatus md_result_lpr(MDResultHandle, const MDLprItem** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_plate(MDResultHandle, size_t i, const char** plate, const char** color);
/* 车牌 4 角点（x1,y1,x2,y2,x3,y3,x4,y4，MDPoint 数组） */
MD_CAPI_EXPORT MDStatus md_result_lpr_keypoints(MDResultHandle, size_t i, const MDPoint** kps, size_t* n);

/* 行人属性 */
MD_CAPI_EXPORT MDStatus md_result_attribute(MDResultHandle, const MDAttrItem** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_attr_scores(MDResultHandle, size_t i, const float** scores, size_t* n);

/* 年龄 / 性别（单值，0 号索引；批量句柄下读 index 0） */
MD_CAPI_EXPORT MDStatus md_result_age(MDResultHandle, int* age);
MD_CAPI_EXPORT MDStatus md_result_gender(MDResultHandle, int* gender);
/* 年龄 / 性别 批量：平铺 int 数组（每图一个值） */
MD_CAPI_EXPORT MDStatus md_result_age_batch(MDResultHandle, const int** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_gender_batch(MDResultHandle, const int** items, size_t* count);

/* 人脸防伪（每实例 label：0=REAL, 1=FUZZY, 2=SPOOF） */
MD_CAPI_EXPORT MDStatus md_result_spoof(MDResultHandle, size_t i, int* label);

/* ==================== 2D 批量结果 getter（按图索引，逐图取项数组） ====================
 * 批量结果来自 md_model_predict_batch（每图一组）。用法：先调 *_batch 数组 getter 取该图项数组，
 * 再（如需）用 (图,项) 版 getter 读 kps/mask/embedding/plate/ocr 行。返回指针在结果句柄存活期内稳定。
 * 单图（md_model_predict）仍用上方不带 _batch 的 getter，语义不变。 */

/* Detection */
MD_CAPI_EXPORT MDStatus md_result_detection_batch(MDResultHandle, size_t img_i, const MDDetectionItem** items, size_t* count);
/* Classification（每图 top-k） */
MD_CAPI_EXPORT MDStatus md_result_classification_batch(MDResultHandle, size_t img_i, const MDClassifyItem** items, size_t* count);
/* Pose（bbox + 骨架） */
MD_CAPI_EXPORT MDStatus md_result_pose_batch(MDResultHandle, size_t img_i, const MDPoseItem** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_keypoints_batch(MDResultHandle, size_t img_i, size_t item_j, const MDPoint3** kps, size_t* n);
/* OBB */
MD_CAPI_EXPORT MDStatus md_result_obb_batch(MDResultHandle, size_t img_i, const MDObbItem** items, size_t* count);
/* InstanceSeg（mask 用 (图,项) 版） */
MD_CAPI_EXPORT MDStatus md_result_instance_seg_batch(MDResultHandle, size_t img_i, const MDIsegItem** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_mask_batch(MDResultHandle, size_t img_i, size_t item_j, const unsigned char** buf,
                              size_t* out_h, size_t* out_w);
/* FaceDet（kps 用 (图,项) 版） */
MD_CAPI_EXPORT MDStatus md_result_face_batch(MDResultHandle, size_t img_i, const MDFaceItem** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_face_kps_batch(MDResultHandle, size_t img_i, size_t item_j, const MDPoint** kps, size_t* n);
/* FaceRec（每图一个 embedding） */
MD_CAPI_EXPORT MDStatus md_result_face_embedding_batch(MDResultHandle, size_t img_i, const float** embedding, size_t* emb_n);
/* ReID（每图一个 512-d embedding） */
MD_CAPI_EXPORT MDStatus md_result_reid_embedding_batch(MDResultHandle, size_t img_i, const float** embedding, size_t* emb_n);
/* InsightFace（完整分析，按图） */
MD_CAPI_EXPORT MDStatus md_result_insightface_batch(MDResultHandle, size_t img_i, const MDInsightFaceItem** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_insightface_kps_batch(MDResultHandle, size_t img_i, size_t item_j, const MDPoint** kps, size_t* n);
MD_CAPI_EXPORT MDStatus md_result_insightface_embedding_batch(MDResultHandle, size_t img_i, size_t item_j,
                                              const float** embedding, size_t* emb_n);
MD_CAPI_EXPORT MDStatus md_result_insightface_pose_batch(MDResultHandle, size_t img_i, size_t item_j,
                                         const float** pose, size_t* n);
/* OCR（按 (图,行) 读） */
MD_CAPI_EXPORT MDStatus md_result_ocr_batch(MDResultHandle, size_t img_i, size_t line_j, const int** quad,
                            const char** text, float* score);
MD_CAPI_EXPORT MDStatus md_result_ocr_cls_batch(MDResultHandle, size_t img_i, size_t line_j, int* cls_label, float* cls_score);
/* LPR（plate/keypoints 用 (图,项) 版） */
MD_CAPI_EXPORT MDStatus md_result_lpr_batch(MDResultHandle, size_t img_i, const MDLprItem** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_plate_batch(MDResultHandle, size_t img_i, size_t item_j, const char** plate, const char** color);
MD_CAPI_EXPORT MDStatus md_result_lpr_keypoints_batch(MDResultHandle, size_t img_i, size_t item_j, const MDPoint** kps, size_t* n);
/* 行人属性（scores 用 (图,项) 版） */
MD_CAPI_EXPORT MDStatus md_result_attribute_batch(MDResultHandle, size_t img_i, const MDAttrItem** items, size_t* count);
MD_CAPI_EXPORT MDStatus md_result_attr_scores_batch(MDResultHandle, size_t img_i, size_t item_j, const float** scores, size_t* n);

/* ==================== 多目标跟踪器 ==================== */

typedef enum MD_TRACKER_KIND {
    MD_TRACKER_BYTETRACK = 0,
    MD_TRACKER_BOTSORT   = 1,
    MD_TRACKER_STRONGSORT= 2,
} MDTrackerKind;

/* 跟踪目标的状态（对应 C++ TrackState：New/Tracked/Lost/Removed） */
typedef enum MD_TRACK_STATE {
    MD_TRACK_NEW = 0,
    MD_TRACK_TRACKED = 1,
    MD_TRACK_LOST = 2,
    MD_TRACK_REMOVED = 3
} MDTrackState;

typedef struct MDTrackItem {
    float x, y, w, h;
    int track_id;
    int label_id;
    float score;
    int state;   /* MDTrackState */
} MDTrackItem;

typedef struct md_tracker_handle* MDTrackerHandle;

/* 创建跟踪器（按 kind 选择 ByteTracker / BotSort / StrongSort）。
 * 纯 CPU 无模型依赖；跟踪 ID 跨帧稳定，reset() 归零。 */
MD_CAPI_EXPORT MDStatus md_tracker_create(MDTrackerKind kind, MDTrackerHandle* out);
MD_CAPI_EXPORT void md_tracker_destroy(MDTrackerHandle h);

/* 非变异容量查询：计算对给定检测调用 md_tracker_update(n) 会产生的 TrackResult 数量，
 * 但**不推进**跟踪器状态。调用方应先以本函数确定输出缓冲大小，再分配并用该容量调用
 * md_tracker_update（见下）。返回 MD_OK，*out_count 置为需要数。 */
MD_CAPI_EXPORT MDStatus md_tracker_capacity(MDTrackerHandle h, const MDBox* boxes,
                          const float* scores, const int* label_ids, size_t n,
                          size_t* out_count);

/* 有状态更新提交：推进跟踪器一帧并写入输出。
 * out 指向调用方分配的 MDTrackItem 数组，其容量为 *out_count（进入时）。
 * 返回时 *out_count 置为实际写入数；若容量不足则报 MD_ERR_INVALID_ARGUMENT，
 * 且不写越界（*out_count 置为需要数）。
 *
 * 查询/提交契约（client）：对每一逻辑帧——
 *   1) md_tracker_capacity(..., &need)   —— 非变异，仅查询，不推进状态；
 *   2) 分配 need 个 MDTrackItem；
 *   3) md_tracker_update(..., cap=need)  —— 有状态，推进**一次**。
 * 切勿用 md_tracker_update 自身作容量探测（会双重推进帧状态）。 */
MD_CAPI_EXPORT MDStatus md_tracker_update(MDTrackerHandle h, const MDBox* boxes,
                          const float* scores, const int* label_ids, size_t n,
                          MDTrackItem* out, size_t* out_count);

/* 通用命名参数设置（double 值）。
 * 支持名字：track_thresh / high_thresh / low_thresh / max_age / min_hits /
 *          iou_threshold / match_thresh / ema_alpha / fuse_score_weight /
 *          appearance_priority / with_cmc
 * 未知名 → MD_ERR_INVALID_ARGUMENT；不支持的 kind 只忽略不适用项。 */
MD_CAPI_EXPORT MDStatus md_tracker_set_params(MDTrackerHandle h, const char* name, double value);

MD_CAPI_EXPORT MDStatus md_tracker_reset(MDTrackerHandle h);

/* ==================== 条码 / 二维码识别 ==================== */

/* 单个条码 / 二维码解码结果（blittable，供 C#/Rust 直接镜像） */
typedef struct MD_BarcodeItem {
    char text[256];      /* 解码文本 UTF-8 */
    char format[16];     /* 格式，如 "QR_CODE" */
    float quad[8];       /* 4 点坐标 (x0,y0,x1,y1,x2,y2,x3,y3)，左上起顺时针 */
    float score;
    int32_t is_qr;
} MD_BarcodeItem;

typedef struct md_barcode_handle* MDBarcodeHandle;

/* 创建条码识别器（纯 CV，无模型依赖）。formats 默认 FMT_ALL。 */
MD_CAPI_EXPORT MDStatus md_barcode_create(MDBarcodeHandle* out);
MD_CAPI_EXPORT void md_barcode_destroy(MDBarcodeHandle h);

/* 限定解码格式子集（FMT_* 位或，见 C++ Formats） */
MD_CAPI_EXPORT MDStatus md_barcode_set_formats(MDBarcodeHandle h, uint32_t formats);

/* 检测并解码图片中的所有码。
 * 容量查询：items==nullptr 时置 *count 为需要数（不写入），返回 MD_OK；
 * 否则 *count 进入时为 items 数组容量，返回时置为实际写入数（写入 min(cap,need) 条）。 */
MD_CAPI_EXPORT MDStatus md_barcode_detect(MDBarcodeHandle h, MDImageHandle img,
                                          MD_BarcodeItem* items, uint32_t* count);

/* ==================== 绘制（对 MDImageHandle 就地绘制） ==================== */

MD_CAPI_EXPORT MDStatus md_draw_rect(MDImageHandle, float x, float y, float w, float h,
                      MDColorRGBA color, float alpha);
MD_CAPI_EXPORT MDStatus md_draw_polygon(MDImageHandle, const float* xs, const float* ys, size_t n,
                         MDColorRGBA color, float alpha);
MD_CAPI_EXPORT MDStatus md_draw_text(MDImageHandle, float x, float y, const char* text,
                      const char* font_path, int font_size, MDColorRGBA color, float alpha);

/* ==================== 结果可视化（直接复用 C++ 的 vis_* 系列） ==================== */

/* 类别名映射：label_id -> 名称 */
typedef struct MDLabelItem {
    int id;
    const char* name;
} MDLabelItem;

/* 绘制选项（与 C++ vis_* 参数一一对应，第三方可完全控制） */
typedef struct MDDrawOptions {
    double threshold;            /* 置信度阈值（默认 0.5） */
    const MDLabelItem* label_map; /* 类别名映射，可为 NULL */
    size_t label_map_size;       /* label_map 条数 */
    const char* font_path;       /* 字体文件路径，可为 NULL */
    int font_size;               /* 默认 14 */
    double alpha;                /* 半透明混合系数（默认 0.15） */
    int save_result;             /* 非 0 保存 vis_result.jpg */
} MDDrawOptions;

/* 把预测结果就地绘制到图像上（内部调用 C++ vis_det/vis_obb/vis_pose/vis_ocr/...）。
 * 支持的 kind：detection / obb / pose / keypoints / instance_seg / sem_seg / depth /
 *               ocr / lpr / attr / classification；其余返回 MD_ERR_UNSUPPORTED_TYPE。
 * 注意：绘制需在结果句柄存活期间调用（Predict 返回的结果对象持有句柄，绘制前请勿释放）。 */
MD_CAPI_EXPORT MDStatus md_draw_result(MDImageHandle, MDResultHandle, const MDDrawOptions* opt);

/* ==================== 解决方案（vision::solution/tool） ==================== */

typedef enum MD_SOLUTION_KIND {
    MD_SOLUTION_OBJECT_COUNTER = 0,
    MD_SOLUTION_HEATMAP,
    MD_SOLUTION_SPEED,
    MD_SOLUTION_DISTANCE,
    MD_SOLUTION_WORKOUT,
    MD_SOLUTION_PARKING,
} MDSolutionKind;

MD_CAPI_EXPORT MDStatus md_solution_create(MDSolutionHandle* out, MDSolutionKind kind);
MD_CAPI_EXPORT MDStatus md_solution_destroy(MDSolutionHandle);
MD_CAPI_EXPORT MDStatus md_solution_object_counter_set_line(MDSolutionHandle h, float ax, float ay, float bx, float by);
MD_CAPI_EXPORT MDStatus md_solution_object_counter_update(MDSolutionHandle h, const float* boxes, size_t n,
                                                          const int* label_ids, const int* track_ids);
MD_CAPI_EXPORT MDStatus md_solution_object_counter_hline(MDSolutionHandle h, int* in, int* out_count);
MD_CAPI_EXPORT MDStatus md_solution_heatmap_set_size(MDSolutionHandle h, int w, int hh);
MD_CAPI_EXPORT MDStatus md_solution_heatmap_update(MDSolutionHandle h, const float* boxes, size_t n,
                                                   int frame_w, int frame_h);
MD_CAPI_EXPORT MDStatus md_solution_heatmap_peak(MDSolutionHandle h, int* x, int* y);
MD_CAPI_EXPORT MDStatus md_vision_iou4(float ax, float ay, float aw, float ah,
                                       float bx, float by, float bw, float bh, float* out);

#ifdef __cplusplus
}
#endif

#endif /* MD_CAPI_H */
