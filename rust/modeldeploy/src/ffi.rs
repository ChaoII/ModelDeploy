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

#[repr(C)]
#[derive(Debug)]
pub struct MDMapData {
    pub data: *mut MDMapNode,
    pub size: c_int,
}

#[repr(C)]
#[derive(Debug)]
pub struct MDOCRModelParameters {
    pub det_model: *const c_char,
    pub cls_model: *const c_char,
    pub rec_model: *const c_char,
    pub rec_dict_path: *const c_char,
    pub db_thresh: c_float,
    pub db_box_thresh: c_float,
    pub db_unclip_ratio: c_float,
    pub db_score_mode: *const c_char,
}

#[repr(C)]
#[derive(Debug)]
pub struct MDKokoroParameters {
    pub model: *mut c_char,
    pub tokens: *mut c_char,
    pub lexicons: *mut c_char,
    pub voice: *mut c_char,
    pub jieba: *mut c_char,
}

// ════════════════════════════════════════════════════════════════
// extern "C" 函数声明
// ════════════════════════════════════════════════════════════════

extern "C" {

    // ── RuntimeOption ──

    pub fn md_create_default_runtime_option() -> MDRuntimeOption;

    // ── Image ──

    pub fn md_read_image(path: *const c_char, image: *mut MDImage) -> MDStatusCode;
    pub fn md_save_image(image: *const MDImage, path: *const c_char) -> MDStatusCode;
    pub fn md_clone_image(src: *const MDImage, dst: *mut MDImage) -> MDStatusCode;
    pub fn md_free_image(image: *mut MDImage);
    pub fn md_nv12_to_bgr24(
        y_plane: *const u8,
        uv_plane: *const u8,
        width: c_int,
        height: c_int,
        y_step: c_int,
        uv_step: c_int,
        image: *mut MDImage,
    ) -> MDStatusCode;

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
    ) -> MDStatusCode;

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
    pub fn md_set_face_det_input_size(model: *const MDModel, size: MDSize) -> MDStatusCode;

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
}
