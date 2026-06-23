//! ModelDeploy C API 的 Rust FFI 绑定（手动定义）
//!
//! 对应 CAPI 头文件: capi/common/md_types.h, capi/vision/detection/detection_capi.h 等
//! 所有函数均为 `extern "C"` 调用，通过 link 到 ModelDeploySDK 动态库

#![allow(non_camel_case_types, dead_code, non_upper_case_globals)]

use libc::{c_char, c_float, c_int, c_void};

// ════════════════════════════════════════════════════════════════
// 枚举（repr(transparent) + 常量）
// ════════════════════════════════════════════════════════════════

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MDStatusCode(pub i32);
pub const MDStatusCode_Success: MDStatusCode = MDStatusCode(0x00);
pub const MDStatusCode_PathNotFound: MDStatusCode = MDStatusCode(0x01);
pub const MDStatusCode_FileOpenFailed: MDStatusCode = MDStatusCode(0x02);
pub const MDStatusCode_CallError: MDStatusCode = MDStatusCode(0x03);
pub const MDStatusCode_ModelInitializeFailed: MDStatusCode = MDStatusCode(0x04);
pub const MDStatusCode_ModelPredictFailed: MDStatusCode = MDStatusCode(0x05);
pub const MDStatusCode_MemoryAllocatedFailed: MDStatusCode = MDStatusCode(0x06);
pub const MDStatusCode_ModelTypeError: MDStatusCode = MDStatusCode(0x07);
pub const MDStatusCode_WriteWaveFailed: MDStatusCode = MDStatusCode(0x08);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MDModelType(pub i32);
pub const MDModelType_Classification: MDModelType = MDModelType(0);
pub const MDModelType_Detection: MDModelType = MDModelType(1);
pub const MDModelType_Keypoint: MDModelType = MDModelType(2);
pub const MDModelType_OCR: MDModelType = MDModelType(3);
pub const MDModelType_FACE: MDModelType = MDModelType(4);
pub const MDModelType_LPR: MDModelType = MDModelType(5);
pub const MDModelType_PIPELINE: MDModelType = MDModelType(6);
pub const MDModelType_ASR: MDModelType = MDModelType(7);
pub const MDModelType_TTS: MDModelType = MDModelType(8);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MDModelFormat(pub i32);
pub const MDModelFormat_ONNX: MDModelFormat = MDModelFormat(0);
pub const MDModelFormat_MNN: MDModelFormat = MDModelFormat(1);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MDDevice(pub i32);
pub const MD_DEVICE_CPU: MDDevice = MDDevice(0);
pub const MD_DEVICE_GPU: MDDevice = MDDevice(1);
pub const MD_DEVICE_OPENCL: MDDevice = MDDevice(2);
pub const MD_DEVICE_VULKAN: MDDevice = MDDevice(3);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MDBackend(pub i32);
pub const MD_BACKEND_ORT: MDBackend = MDBackend(0);
pub const MD_BACKEND_MNN: MDBackend = MDBackend(1);
pub const MD_BACKEND_TRT: MDBackend = MDBackend(2);
pub const MD_BACKEND_NONE: MDBackend = MDBackend(3);

// ════════════════════════════════════════════════════════════════
// 结构体
// ════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDPoint {
    pub x: c_int,
    pub y: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDPoint3f {
    pub x: c_float,
    pub y: c_float,
    pub z: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDRect {
    pub x: c_int,
    pub y: c_int,
    pub width: c_int,
    pub height: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDRotatedRect {
    pub xc: c_float,
    pub yc: c_float,
    pub width: c_float,
    pub height: c_float,
    pub angle: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDSize {
    pub width: c_int,
    pub height: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[repr(C)]
#[derive(Debug)]
pub struct MDPolygon {
    pub data: *mut MDPoint,
    pub size: c_int,
}

#[repr(C)]
#[derive(Debug)]
pub struct MDMask {
    pub buffer: *mut c_char,
    pub buffer_size: c_int,
    pub shape: *mut c_int,
    pub num_dims: c_int,
}

/// 图像
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MDImage {
    pub width: c_int,
    pub height: c_int,
    pub channels: c_int,
    pub data: *mut u8,
}

/// 模型句柄（所有模型类型的通用结构）
#[repr(C)]
#[derive(Debug)]
pub struct MDModel {
    pub model_name: *mut c_char,
    pub type_: MDModelType,
    pub format: MDModelFormat,
    pub model_content: *mut c_void,
}

/// 运行时选项
#[repr(C)]
#[derive(Debug)]
pub struct MDRuntimeOption {
    pub trt_min_shape: *const c_char,
    pub trt_opt_shape: *const c_char,
    pub trt_max_shape: *const c_char,
    pub trt_engine_cache_path: *const c_char,
    pub enable_fp16: c_int,
    pub cpu_thread_num: c_int,
    pub device_id: c_int,
    pub enable_trt: c_int,
    pub device: MDDevice,
    pub backend: MDBackend,
    pub graph_opt_level: c_int,
    pub password: *const c_char,
    pub ort_log_severity: c_int,
}

// ════════════════════════════════════════════════════════════════
// 结果类型
// ════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Debug)]
pub struct MDDetectionResult {
    pub box_: MDRect,
    pub label_id: c_int,
    pub score: c_float,
}

#[repr(C)]
#[derive(Debug)]
pub struct MDDetectionResults {
    pub data: *mut MDDetectionResult,
    pub size: c_int,
}

#[repr(C)]
#[derive(Debug)]
pub struct MDClassificationResult {
    pub label_id: c_int,
    pub score: c_float,
}

#[repr(C)]
#[derive(Debug)]
pub struct MDClassificationResults {
    pub data: *mut MDClassificationResult,
    pub size: c_int,
}

#[repr(C)]
#[derive(Debug)]
pub struct MDKeyPointResult {
    pub box_: MDRect,
    pub score: c_float,
    pub label_id: c_int,
    pub keypoints: *mut MDPoint3f,
    pub keypoints_size: c_int,
}

#[repr(C)]
#[derive(Debug)]
pub struct MDKeyPointResults {
    pub data: *mut MDKeyPointResult,
    pub size: c_int,
}

#[repr(C)]
#[derive(Debug)]
pub struct MDTTSResult {
    pub data: *mut c_float,
    pub size: c_int,
    pub sample_rate: c_int,
}

#[repr(C)]
#[derive(Debug)]
pub struct MDImageResults {
    pub data: *mut MDImage,
    pub size: c_int,
}

#[repr(C)]
#[derive(Debug)]
pub struct MDMapNode {
    pub key: *mut c_char,
    pub value: *mut c_char,
}

// ── Kokoro TTS 参数 ──
#[repr(C)]
#[derive(Debug)]
pub struct MDKokoroParameters {
    pub model: *mut c_char,
    pub tokens: *mut c_char,
    pub lexicons: *mut c_char,
    pub voice: *mut c_char,
    pub jieba: *mut c_char,
}

// ── OBB ──
#[repr(C)]
#[derive(Debug)]
pub struct MDObbResult {
    pub rotated_box: MDRotatedRect,
    pub label_id: c_int,
    pub score: c_float,
}
#[repr(C)]
#[derive(Debug)]
pub struct MDObbResults {
    pub data: *mut MDObbResult,
    pub size: c_int,
}

// ── 实例分割 ──
#[repr(C)]
#[derive(Debug)]
pub struct MDIsegResult {
    pub box_: MDRect,
    pub mask: MDMask,
    pub label_id: c_int,
    pub score: c_float,
}
#[repr(C)]
#[derive(Debug)]
pub struct MDIsegResults {
    pub data: *mut MDIsegResult,
    pub size: c_int,
}

// ── OCR ──
#[repr(C)]
#[derive(Debug)]
pub struct MDOCRResult {
    pub box_: MDPolygon,
    pub text: *mut c_char,
    pub score: c_float,
    pub table_boxes: MDPolygon,
    pub table_structure: *mut c_char,
}
#[repr(C)]
#[derive(Debug)]
pub struct MDOCRResults {
    pub data: *mut MDOCRResult,
    pub table_html: *mut c_char,
    pub size: c_int,
}

// ── LPR ──
#[repr(C)]
#[derive(Debug)]
pub struct MDLPRResult {
    pub box_: MDRect,
    pub landmarks: *mut MDPoint,
    pub landmarks_size: c_int,
    pub label_id: c_int,
    pub score: c_float,
    pub car_plate_str: *mut c_char,
    pub car_plate_color: *mut c_char,
}
#[repr(C)]
#[derive(Debug)]
pub struct MDLPRResults {
    pub data: *mut MDLPRResult,
    pub size: c_int,
}

// ── 行人属性 ──
#[repr(C)]
#[derive(Debug)]
pub struct MDAttributeResult {
    pub box_: MDRect,
    pub box_label_id: c_int,
    pub box_score: c_float,
    pub attr_scores: *mut c_float,
    pub attr_scores_size: c_int,
}
#[repr(C)]
#[derive(Debug)]
pub struct MDAttributeResults {
    pub data: *mut MDAttributeResult,
    pub size: c_int,
}

// ── 人脸识别 ──
#[repr(C)]
#[derive(Debug)]
pub struct MDFaceRecognizerResult {
    pub embedding: *mut c_float,
    pub size: c_int,
}
#[repr(C)]
#[derive(Debug)]
pub struct MDFaceRecognizerResults {
    pub data: *mut MDFaceRecognizerResult,
    pub size: c_int,
}

// ── 人脸防伪第二段 ──
#[repr(C)]
#[derive(Debug)]
pub struct MDFaceAsSecondResult {
    pub label_id: c_int,
    pub score: c_float,
}
#[repr(C)]
#[derive(Debug)]
pub struct MDFaceAsSecondResults {
    pub data: *mut MDFaceAsSecondResult,
    pub size: c_int,
}

// ── 人脸防伪 Pipeline ──
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct MDFaceAsResult(pub i32);
pub const MDFaceAsResult_REAL: MDFaceAsResult = MDFaceAsResult(0);
pub const MDFaceAsResult_FUZZY: MDFaceAsResult = MDFaceAsResult(1);
pub const MDFaceAsResult_SPOOF: MDFaceAsResult = MDFaceAsResult(2);

#[repr(C)]
#[derive(Debug)]
pub struct MDFaceAsResults {
    pub data: *mut MDFaceAsResult,
    pub size: c_int,
}

// ── 人脸年龄/性别 ──
pub type MDFaceAgeResult = c_int;
pub const MDFACE_AGE_UNKNOWN: c_int = -1;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MDFaceGenderResult(pub i32);
pub const MDFaceGenderResult_FEMALE: MDFaceGenderResult = MDFaceGenderResult(0);
pub const MDFaceGenderResult_MALE: MDFaceGenderResult = MDFaceGenderResult(1);

// ── OCR 模型参数 ──
#[repr(C)]
#[derive(Debug)]
pub struct MDOCRModelParameters {
    pub det_model_file: *const c_char,
    pub cls_model_file: *const c_char,
    pub rec_model_file: *const c_char,
    pub dict_path: *const c_char,
    pub max_side_len: c_int,
    pub det_db_thresh: f64,
    pub det_db_box_thresh: f64,
    pub det_db_unclip_ratio: f64,
    pub det_db_score_mode: *const c_char,
    pub use_dilation: c_int,
    pub rec_batch_size: c_int,
}

// ── Structure Table 模型参数 ──
#[repr(C)]
#[derive(Debug)]
pub struct MDStructureTableModelParameters {
    pub det_model_file: *const c_char,
    pub rec_model_file: *const c_char,
    pub table_model_file: *const c_char,
    pub rec_label_file: *const c_char,
    pub table_char_dict_path: *const c_char,
    pub max_side_len: c_int,
    pub det_db_thresh: f64,
    pub det_db_box_thresh: f64,
    pub det_db_unclip_ratio: f64,
    pub det_db_score_mode: *const c_char,
    pub use_dilation: c_int,
    pub rec_batch_size: c_int,
}

// ════════════════════════════════════════════════════════════════
// extern "C" 函数声明
// ════════════════════════════════════════════════════════════════

extern "C" {

    // ── RuntimeOption ──

    pub fn md_create_default_runtime_option() -> MDRuntimeOption;

    // ── Image ──

    pub fn md_read_image(path: *const c_char) -> MDImage;
    pub fn md_clone_image(src: *const MDImage) -> MDImage;
    pub fn md_save_image(image: *const MDImage, path: *const c_char);
    pub fn md_free_image(image: *mut MDImage);
    pub fn md_from_nv12_data_to_bgr24(
        data: *const u8, width: c_int, height: c_int,
    ) -> MDImage;

    // ── 工具 ──

    pub fn md_get_version() -> *const c_char;

    // ── 检测（Detection） ──

    pub fn md_create_detection_model(
        model: *mut MDModel,
        path: *const c_char,
        option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_detection_predict(
        model: *const MDModel,
        image: *const MDImage,
        results: *mut MDDetectionResults,
    ) -> MDStatusCode;
    pub fn md_free_detection_result(results: *mut MDDetectionResults);
    pub fn md_free_detection_model(model: *mut MDModel);
    pub fn md_set_detection_input_size(model: *const MDModel, size: MDSize) -> MDStatusCode;
    pub fn md_draw_detection_result(
        image: *const MDImage,
        results: *mut MDDetectionResults,
        threshold: f64,
        font_path: *const c_char,
        out: *mut MDImage,
    );

    // ── 分类（Classification） ──

    pub fn md_create_classification_model(
        model: *mut MDModel,
        path: *const c_char,
        option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_classification_predict(
        model: *const MDModel,
        image: *const MDImage,
        topk: c_int,
        results: *mut MDClassificationResults,
    ) -> MDStatusCode;
    pub fn md_free_classification_result(results: *mut MDClassificationResults);
    pub fn md_free_classification_model(model: *mut MDModel);

    // ── 人脸检测（Face Detection） ──

    pub fn md_create_face_det_model(
        model: *mut MDModel,
        path: *const c_char,
        option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_face_det_predict(
        model: *const MDModel,
        image: *const MDImage,
        results: *mut MDKeyPointResults,
    ) -> MDStatusCode;
    pub fn md_free_face_det_result(results: *mut MDKeyPointResults);
    pub fn md_free_face_det_model(model: *mut MDModel);

    // ── TTS（Kokoro） ──

    pub fn md_create_kokoro_model(
        model: *mut MDModel,
        params: *const MDKokoroParameters,
        option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_kokoro_model_predict(
        model: *const MDModel,
        text: *const c_char,
        result: *mut MDTTSResult,
    ) -> MDStatusCode;
    pub fn md_free_kokoro_result(result: *mut MDTTSResult);
    pub fn md_free_kokoro_model(model: *mut MDModel);
    pub fn md_write_wav(result: *const MDTTSResult, path: *const c_char) -> MDStatusCode;

    // ── OBB ──
    pub fn md_create_obb_model(
        model: *mut MDModel, path: *const c_char, option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_set_obb_input_size(model: *const MDModel, size: MDSize) -> MDStatusCode;
    pub fn md_obb_predict(
        model: *const MDModel, image: *const MDImage, results: *mut MDObbResults,
    ) -> MDStatusCode;
    pub fn md_free_obb_result(results: *mut MDObbResults);
    pub fn md_free_obb_model(model: *mut MDModel);

    // ── 实例分割 ──
    pub fn md_create_instance_seg_model(
        model: *mut MDModel, path: *const c_char, option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_set_instance_seg_input_size(model: *const MDModel, size: MDSize) -> MDStatusCode;
    pub fn md_instance_seg_predict(
        model: *const MDModel, image: *const MDImage, results: *mut MDIsegResults,
    ) -> MDStatusCode;
    pub fn md_free_instance_seg_result(results: *mut MDIsegResults);
    pub fn md_free_instance_seg_model(model: *mut MDModel);

    // ── OCR ──
    pub fn md_create_ocr_model(
        model: *mut MDModel, params: *const MDOCRModelParameters, option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_ocr_model_predict(
        model: *const MDModel, image: *mut MDImage, results: *mut MDOCRResults,
    ) -> MDStatusCode;
    pub fn md_free_ocr_result(results: *mut MDOCRResults);
    pub fn md_free_ocr_model(model: *mut MDModel);

    // ── OCR Recognition ──
    pub fn md_create_ocr_recognition_model(
        model: *mut MDModel, path: *const c_char, dict_path: *const c_char,
        option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_ocr_recognition_model_predict(
        model: *const MDModel, image: *const MDImage, result: *mut MDOCRResult,
    ) -> MDStatusCode;
    pub fn md_free_ocr_recognition_result(result: *mut MDOCRResult);
    pub fn md_free_ocr_recognition_model(model: *mut MDModel);

    // ── Structure Table ──
    pub fn md_create_structure_table_model(
        model: *mut MDModel, params: *const MDStructureTableModelParameters,
        option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_structure_table_model_predict(
        model: *const MDModel, image: *mut MDImage, results: *mut MDOCRResults,
    ) -> MDStatusCode;
    pub fn md_free_structure_table_result(results: *mut MDOCRResults);
    pub fn md_free_structure_table_model(model: *mut MDModel);

    // ── 姿态估计（Keypoint） ──
    pub fn md_create_keypoint_model(
        model: *mut MDModel, path: *const c_char, option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_set_keypoint_input_size(model: *const MDModel, size: MDSize) -> MDStatusCode;
    pub fn md_set_keypoint_num(model: *const MDModel, num: c_int) -> MDStatusCode;
    pub fn md_keypoint_predict(
        model: *const MDModel, image: *const MDImage, results: *mut MDKeyPointResults,
    ) -> MDStatusCode;
    pub fn md_free_keypoint_result(results: *mut MDKeyPointResults);
    pub fn md_free_keypoint_model(model: *mut MDModel);

    // ── LPR 检测 ──
    pub fn md_create_lpr_det_model(
        model: *mut MDModel, path: *const c_char, option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_lpr_det_predict(
        model: *const MDModel, image: *const MDImage, results: *mut MDKeyPointResults,
    ) -> MDStatusCode;
    pub fn md_free_lpr_det_result(results: *mut MDKeyPointResults);
    pub fn md_free_lpr_det_model(model: *mut MDModel);

    // ── LPR 识别 ──
    pub fn md_create_lpr_rec_model(
        model: *mut MDModel, path: *const c_char, option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_lpr_rec_predict(
        model: *const MDModel, image: *const MDImage, results: *mut MDLPRResults,
    ) -> MDStatusCode;
    pub fn md_free_lpr_rec_result(results: *mut MDLPRResults);
    pub fn md_free_lpr_rec_model(model: *mut MDModel);

    // ── LPR Pipeline ──
    pub fn md_create_lpr_pipeline_model(
        model: *mut MDModel, det_path: *const c_char, rec_path: *const c_char,
        option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_lpr_pipeline_predict(
        model: *const MDModel, image: *const MDImage, results: *mut MDLPRResults,
    ) -> MDStatusCode;
    pub fn md_free_lpr_pipeline_result(results: *mut MDLPRResults);
    pub fn md_free_lpr_pipeline_model(model: *mut MDModel);

    // ── 行人属性 ──
    pub fn md_create_attr_model(
        model: *mut MDModel, det_path: *const c_char, cls_path: *const c_char,
        option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_attr_model_predict(
        model: *const MDModel, image: *const MDImage, results: *mut MDAttributeResults,
    ) -> MDStatusCode;
    pub fn md_free_attr_result(results: *mut MDAttributeResults);
    pub fn md_free_attr_model(model: *mut MDModel);

    // ── 人脸识别 ──
    pub fn md_create_face_rec_model(
        model: *mut MDModel, path: *const c_char, option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_face_rec_predict(
        model: *const MDModel, image: *const MDImage, result: *mut MDFaceRecognizerResult,
    ) -> MDStatusCode;
    pub fn md_free_face_rec_result(result: *mut MDFaceRecognizerResult);
    pub fn md_free_face_rec_model(model: *mut MDModel);

    // ── 人脸年龄 ──
    pub fn md_create_face_age_model(
        model: *mut MDModel, path: *const c_char, option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_face_age_predict(
        model: *const MDModel, image: *const MDImage, result: *mut c_int,
    ) -> MDStatusCode;
    pub fn md_free_face_age_result(result: *mut c_int);
    pub fn md_free_face_age_model(model: *mut MDModel);

    // ── 人脸性别 ──
    pub fn md_create_face_gender_model(
        model: *mut MDModel, path: *const c_char, option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_face_gender_predict(
        model: *const MDModel, image: *const MDImage, result: *mut MDFaceGenderResult,
    ) -> MDStatusCode;
    pub fn md_free_face_gender_result(result: *mut MDFaceGenderResult);
    pub fn md_free_face_gender_model(model: *mut MDModel);

    // ── 人脸防伪第一段 ──
    pub fn md_create_face_as_first_model(
        model: *mut MDModel, path: *const c_char, option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_face_as_first_predict(
        model: *const MDModel, image: *const MDImage, result: *mut c_float,
    ) -> MDStatusCode;
    pub fn md_free_face_as_first_model(model: *mut MDModel);

    // ── 人脸防伪第二段 ──
    pub fn md_create_face_as_second_model(
        model: *mut MDModel, path: *const c_char, option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_face_as_second_predict(
        model: *const MDModel, image: *const MDImage, results: *mut MDFaceAsSecondResults,
    ) -> MDStatusCode;
    pub fn md_free_face_as_second_result(results: *mut MDFaceAsSecondResults);
    pub fn md_free_face_as_second_model(model: *mut MDModel);

    // ── 人脸防伪 Pipeline ──
    pub fn md_create_face_as_pipeline_model(
        model: *mut MDModel, det_path: *const c_char, first_path: *const c_char,
        second_path: *const c_char, option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_face_as_pipeline_predict(
        model: *const MDModel, image: *const MDImage, results: *mut MDFaceAsResults,
        fuse_threshold: c_float, clarity_threshold: c_float,
    ) -> MDStatusCode;
    pub fn md_free_face_as_pipeline_result(results: *mut MDFaceAsResults);
    pub fn md_free_face_as_pipeline_model(model: *mut MDModel);

    // ── 人脸识别 Pipeline ──
    pub fn md_create_face_rec_pipeline_model(
        model: *mut MDModel, det_path: *const c_char, rec_path: *const c_char,
        option: *const MDRuntimeOption,
    ) -> MDStatusCode;
    pub fn md_face_rec_pipeline_predict(
        model: *const MDModel, image: *const MDImage, results: *mut MDFaceRecognizerResults,
    ) -> MDStatusCode;
    pub fn md_free_face_rec_pipeline_result(results: *mut MDFaceRecognizerResults);
    pub fn md_free_face_rec_pipeline_model(model: *mut MDModel);
}
