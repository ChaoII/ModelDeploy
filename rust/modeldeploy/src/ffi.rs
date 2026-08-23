//! ModelDeploy C API v2 的 Rust FFI 绑定（手动定义）
//!
//! 对应 CAPI 头文件: capi/md_capi.h
//! 设计: 不透明句柄（MDModelHandle/MDImageHandle/MDResultHandle/MDOptionHandle = 裸指针）
//! 所有函数均为 `extern "C"` 调用，通过 link 到 ModelDeploySDK 动态库

#![allow(non_camel_case_types, dead_code, non_upper_case_globals)]

use libc::{c_char, c_float, c_int, c_uint, c_void};

// ════════════════════════════════════════════════════════════════
// 枚举
// ════════════════════════════════════════════════════════════════

/// 状态码（MDStatus）
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MDStatus {
    OK = 0,
    ERR_NULL_POINTER,
    ERR_INVALID_ARGUMENT,
    ERR_PATH_NOT_FOUND,
    ERR_MODEL_LOAD,
    ERR_MODEL_PREDICT,
    ERR_MODEL_INIT,
    ERR_UNSUPPORTED_TYPE,
    ERR_UNSUPPORTED_BACKEND,
    ERR_OUT_OF_MEMORY,
    ERR_IMAGE_DECODE,
    ERR_BUSY,
    ERR_NOT_IMPLEMENTED,
    ERR_AUDIO_DECODE,
    ERR_INVALID_TYPE,
}

/// 模型类型（MDModelKind）
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MDModelKind {
    DETECTION = 0,
    CLASSIFICATION,
    POSE,
    OBB,
    INSTANCE_SEG,
    SEM_SEG,
    DEPTH,
    FACE_DET,
    FACE_REC,
    FACE_AGE,
    FACE_GENDER,
    FACE_AS,
    FACE_AS_PIPELINE,
    FACE_REC_PIPELINE,
    INSIGHTFACE,
    INSIGHTFACE_DET,
    OCR,
    OCR_DET,
    OCR_REC,
    OCR_CLS,
    LPR_DET,
    LPR_REC,
    LPR_PIPELINE,
    PED_ATTR,
    ASR,
    TTS,
    FACE_AS_SECOND,
    HAND,
    REID,
    SPEAKER_VERIFY,
    FORMULA_RECOGNIZER,
    // CAPI 在 FORMULA_RECOGNIZER(=30) 与 VEHICLE_KEYPOINT 之间还有 TSN(=31)/ST_GCN(=32)，
    // 此处仅包装关键点模型，故显式对齐 CAPI 数值。
    VEHICLE_KEYPOINT = 33,
    FACE_LANDMARK = 34,
}

/// 结果类型（MDResultKind）
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MDResultKind {
    DETECTION = 0,
    CLASSIFICATION,
    POSE,
    OBB,
    INSTANCE_SEG,
    SEM_SEG,
    DEPTH,
    FACE,
    FACE_REC,
    INSIGHTFACE,
    OCR,
    LPR,
    ATTR,
    AGE,
    GENDER,
    ASR,
    TTS,
}

/// 设备（MDDevice）
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MDDevice {
    CPU = 0,
    GPU = 1,
    TPU = 2,
    OPENCL = 3,
    VULKAN = 4,
}

/// 后端（MDBackend）
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MDBackend {
    ORT = 0,
    MNN = 1,
    TRT = 2,
    SOPHGO = 3,
}

// ════════════════════════════════════════════════════════════════
// 句柄（不透明指针）
// ════════════════════════════════════════════════════════════════

pub type MDModelHandle = *mut c_void;
pub type MDImageHandle = *mut c_void;
pub type MDResultHandle = *mut c_void;
pub type MDOptionHandle = *mut c_void;
pub type MDTrackerHandle = *mut c_void;
pub type MDBarcodeHandle = *mut c_void;

// ════════════════════════════════════════════════════════════════
// 通用几何 / 颜色 / blittable 结构
// ════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDBox {
    pub x: c_float,
    pub y: c_float,
    pub w: c_float,
    pub h: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDPoint {
    pub x: c_float,
    pub y: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDPoint3 {
    pub x: c_float,
    pub y: c_float,
    pub z: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDRotatedBox {
    pub cx: c_float,
    pub cy: c_float,
    pub w: c_float,
    pub h: c_float,
    pub angle: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDColorRGBA {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

/// 结果项结构（blittable，数组式 getter 返回其数组）
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDDetectionItem {
    pub x: c_float,
    pub y: c_float,
    pub w: c_float,
    pub h: c_float,
    pub score: c_float,
    pub label_id: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDClassifyItem {
    pub label_id: c_int,
    pub score: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDPoseItem {
    pub x: c_float,
    pub y: c_float,
    pub w: c_float,
    pub h: c_float,
    pub score: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDObbItem {
    pub cx: c_float,
    pub cy: c_float,
    pub w: c_float,
    pub h: c_float,
    pub angle: c_float,
    pub score: c_float,
    pub label_id: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDIsegItem {
    pub x: c_float,
    pub y: c_float,
    pub w: c_float,
    pub h: c_float,
    pub score: c_float,
    pub label_id: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDLprItem {
    pub x: c_float,
    pub y: c_float,
    pub w: c_float,
    pub h: c_float,
    pub score: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDAttrItem {
    pub x: c_float,
    pub y: c_float,
    pub w: c_float,
    pub h: c_float,
    pub box_score: c_float,
    pub box_label_id: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDFaceItem {
    pub x: c_float,
    pub y: c_float,
    pub w: c_float,
    pub h: c_float,
    pub score: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDInsightFaceItem {
    pub x: c_float,
    pub y: c_float,
    pub w: c_float,
    pub h: c_float,
    pub score: c_float,
    pub gender: c_int,
    pub age: c_int,
}

/// 跟踪目标项（对应 capi MDTrackItem）
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDTrackItem {
    pub x: c_float,
    pub y: c_float,
    pub w: c_float,
    pub h: c_float,
    pub track_id: c_int,
    pub label_id: c_int,
    pub score: c_float,
    pub state: c_int, // MDTrackState
}

// ════════════════════════════════════════════════════════════════
// 绘制选项（MDDrawOptions + MDLabelItem）
// ════════════════════════════════════════════════════════════════

/// 条码 / 二维码解码结果项（对应 capi MD_BarcodeItem，blittable）
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDBarcodeItem {
    pub text: [c_char; 256],
    pub format: [c_char; 16],
    pub quad: [c_float; 8],
    pub score: c_float,
    pub is_qr: c_int,
}

impl Default for MDBarcodeItem {
    fn default() -> Self {
        Self {
            text: [0; 256],
            format: [0; 16],
            quad: [0.0; 8],
            score: 0.0,
            is_qr: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDLabelItem {
    pub id: c_int,
    pub name: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MDDrawOptions {
    pub threshold: f64,
    pub label_map: *const MDLabelItem,
    pub label_map_size: usize,
    pub font_path: *const c_char,
    pub font_size: c_int,
    pub alpha: f64,
    pub save_result: c_int,
}

// ════════════════════════════════════════════════════════════════
// extern "C" 函数声明
// ════════════════════════════════════════════════════════════════

extern "C" {
    // ── 错误 ──
    pub fn md_get_last_error() -> *const c_char;

    // ── 选项 ──
    pub fn md_option_create(out: *mut MDOptionHandle) -> MDStatus;
    pub fn md_option_destroy(h: MDOptionHandle);
    pub fn md_option_set_device(h: MDOptionHandle, d: MDDevice);
    pub fn md_option_set_backend(h: MDOptionHandle, b: MDBackend);
    pub fn md_option_set_cpu_threads(h: MDOptionHandle, n: c_int);
    pub fn md_option_set_fp16(h: MDOptionHandle, enable: c_int);
    pub fn md_option_set_trt_engine_path(h: MDOptionHandle, path: *const c_char);

    // ── 图像 ──
    pub fn md_image_from_file(out: *mut MDImageHandle, path: *const c_char) -> MDStatus;
    pub fn md_image_from_bgr24(out: *mut MDImageHandle, bgr: *const c_void, w: c_int, h: c_int) -> MDStatus;
    pub fn md_image_from_rgb24(out: *mut MDImageHandle, rgb: *const c_void, w: c_int, h: c_int) -> MDStatus;
    pub fn md_image_from_nv12(out: *mut MDImageHandle, y: *const c_void, uv: *const c_void,
        w: c_int, h: c_int, step_y: c_int, step_uv: c_int) -> MDStatus;
    pub fn md_image_from_nv12_owned(out: *mut MDImageHandle, y: *const c_void, uv: *const c_void,
        w: c_int, h: c_int, step_y: c_int, step_uv: c_int) -> MDStatus;
    pub fn md_image_from_device_nv12(out: *mut MDImageHandle, y: *const c_void, uv: *const c_void,
        w: c_int, h: c_int, step_y: c_int, step_uv: c_int, dev: MDDevice) -> MDStatus;
    pub fn md_image_from_yuv420p(out: *mut MDImageHandle, data: *const c_void, w: c_int, h: c_int) -> MDStatus;
    pub fn md_image_from_encoded(out: *mut MDImageHandle, bytes: *const c_void, n: usize) -> MDStatus;
    pub fn md_image_from_base64(out: *mut MDImageHandle, b64: *const c_char) -> MDStatus;
    pub fn md_image_clone(src: MDImageHandle, out: *mut MDImageHandle) -> MDStatus;
    pub fn md_image_crop(src: MDImageHandle, x: c_int, y: c_int, w: c_int, h: c_int, out: *mut MDImageHandle) -> MDStatus;
    pub fn md_image_show(h: MDImageHandle) -> MDStatus;
    pub fn md_image_save(h: MDImageHandle, path: *const c_char) -> MDStatus;
    pub fn md_image_encode(h: MDImageHandle, ext: *const c_char, buf: *mut *const u8, n: *mut usize) -> MDStatus;
    pub fn md_image_destroy(h: MDImageHandle);
    pub fn md_image_size(h: MDImageHandle, w: *mut c_int, h: *mut c_int) -> MDStatus;
    pub fn md_image_info(h: MDImageHandle, type_: *mut c_int, dev: *mut MDDevice, nplanes: *mut c_int) -> MDStatus;
    // 取 NV12 帧平面指针（dev 返回帧所在设备；仅对 NV12 有效，CPU BGR 图返回 UNSUPPORTED_TYPE）
    pub fn md_image_plane_ptrs(h: MDImageHandle, dev: *mut MDDevice,
        y: *mut *mut c_void, uv: *mut *mut c_void) -> MDStatus;

    // ── 模型 ──
    pub fn md_model_create(out: *mut MDModelHandle, kind: MDModelKind,
        model_path: *const c_char, opt: MDOptionHandle) -> MDStatus;
    pub fn md_model_destroy(h: MDModelHandle);
    pub fn md_model_clone(src: MDModelHandle, out: *mut MDModelHandle) -> MDStatus;
    pub fn md_model_ready(h: MDModelHandle) -> MDStatus;
    pub fn md_model_set_input_size(h: MDModelHandle, w: c_int, h: c_int) -> MDStatus;
    pub fn md_model_set_cls_input_size(h: MDModelHandle, w: c_int, h: c_int) -> MDStatus;
    pub fn md_model_set_cls_batch_size(h: MDModelHandle, batch: c_int) -> MDStatus;
    pub fn md_model_set_rec_batch_size(h: MDModelHandle, batch: c_int) -> MDStatus;
    pub fn md_model_set_rec_image_shape(h: MDModelHandle, c: c_int, h: c_int, w: c_int) -> MDStatus;
    // 模型前/后处理参数（扁平参数名）
    pub fn md_model_set_param_i(model: MDModelHandle, name: *const c_char, value: i64) -> MDStatus;
    pub fn md_model_set_param_d(model: MDModelHandle, name: *const c_char, value: f64) -> MDStatus;
    pub fn md_model_set_param_b(model: MDModelHandle, name: *const c_char, enable: c_int) -> MDStatus;
    pub fn md_model_set_param_s(model: MDModelHandle, name: *const c_char, value: *const c_char) -> MDStatus;
    pub fn md_model_param_names(kind: MDModelKind, names: *mut *const c_char) -> MDStatus;
    pub fn md_model_param_type(kind: MDModelKind, name: *const c_char, type_out: *mut c_char) -> MDStatus;
    pub fn md_model_predict(h: MDModelHandle, img: MDImageHandle, out: *mut MDResultHandle) -> MDStatus;
    pub fn md_model_predict_batch(h: MDModelHandle, imgs: *mut MDImageHandle, n: usize, out: *mut MDResultHandle) -> MDStatus;

    // ── 音频 ──
    pub fn md_audio_asr_wav(h: MDModelHandle, wav: *const c_char, text: *mut *const c_char) -> MDStatus;
    pub fn md_audio_asr(h: MDModelHandle, samples: *const c_float, n: usize, sample_rate: c_int, text: *mut *const c_char) -> MDStatus;
    pub fn md_audio_tts(h: MDModelHandle, text: *const c_char, voice: *const c_char, speed: c_float,
        sample_rate: *mut c_int, audio: *mut *const c_float, audio_n: *mut usize) -> MDStatus;
    pub fn md_audio_speaker_embed(h: MDModelHandle, samples: *const c_float, n: usize,
        embedding: *mut *const c_float, emb_n: *mut usize) -> MDStatus;
    pub fn md_wav_save(samples: *const c_float, n: usize, sample_rate: c_int, path: *const c_char) -> MDStatus;

    // ── 结果 ──
    pub fn md_result_destroy(h: MDResultHandle);
    pub fn md_result_kind(h: MDResultHandle, kind: *mut MDResultKind) -> MDStatus;
    pub fn md_result_count(h: MDResultHandle, count: *mut usize) -> MDStatus;

    // 数组式 getter
    pub fn md_result_detection(h: MDResultHandle, items: *mut *const MDDetectionItem, count: *mut usize) -> MDStatus;
    pub fn md_result_classification(h: MDResultHandle, items: *mut *const MDClassifyItem, count: *mut usize) -> MDStatus;
    pub fn md_result_pose(h: MDResultHandle, items: *mut *const MDPoseItem, count: *mut usize) -> MDStatus;
    pub fn md_result_keypoints(h: MDResultHandle, i: usize, kps: *mut *const MDPoint3, n: *mut usize) -> MDStatus;
    pub fn md_result_obb(h: MDResultHandle, items: *mut *const MDObbItem, count: *mut usize) -> MDStatus;
    pub fn md_result_instance_seg(h: MDResultHandle, items: *mut *const MDIsegItem, count: *mut usize) -> MDStatus;
    pub fn md_result_mask(h: MDResultHandle, i: usize, buf: *mut *const u8, out_h: *mut usize, out_w: *mut usize) -> MDStatus;
    pub fn md_result_sem_seg(h: MDResultHandle, labels: *mut *const u8, out_h: *mut usize, out_w: *mut usize, num_classes: *mut c_int) -> MDStatus;
    pub fn md_result_sem_seg_batch(h: MDResultHandle, i: usize, labels: *mut *const u8, out_h: *mut usize, out_w: *mut usize, num_classes: *mut c_int) -> MDStatus;
    pub fn md_result_depth(h: MDResultHandle, depth: *mut *const c_float, out_h: *mut usize, out_w: *mut usize) -> MDStatus;
    pub fn md_result_depth_batch(h: MDResultHandle, i: usize, depth: *mut *const c_float, out_h: *mut usize, out_w: *mut usize) -> MDStatus;
    pub fn md_result_face(h: MDResultHandle, items: *mut *const MDFaceItem, count: *mut usize) -> MDStatus;
    pub fn md_result_face_kps(h: MDResultHandle, i: usize, kps: *mut *const MDPoint, n: *mut usize) -> MDStatus;
    pub fn md_result_face_embedding(h: MDResultHandle, i: usize, emb: *mut *const c_float, n: *mut usize) -> MDStatus;
    pub fn md_result_reid_embedding(h: MDResultHandle, i: usize, emb: *mut *const c_float, n: *mut usize) -> MDStatus;
    pub fn md_result_insightface(h: MDResultHandle, items: *mut *const MDInsightFaceItem, count: *mut usize) -> MDStatus;
    pub fn md_result_insightface_kps(h: MDResultHandle, i: usize, kps: *mut *const MDPoint, n: *mut usize) -> MDStatus;
    pub fn md_result_insightface_embedding(h: MDResultHandle, i: usize, emb: *mut *const c_float, n: *mut usize) -> MDStatus;
    pub fn md_result_insightface_pose(h: MDResultHandle, i: usize, pose: *mut *const c_float, n: *mut usize) -> MDStatus;
    pub fn md_result_ocr(h: MDResultHandle, i: usize, quad: *mut *const c_int, text: *mut *const c_char, score: *mut c_float) -> MDStatus;
    pub fn md_result_ocr_cls(h: MDResultHandle, i: usize, cls_label: *mut c_int, cls_score: *mut c_float) -> MDStatus;
    pub fn md_result_lpr(h: MDResultHandle, items: *mut *const MDLprItem, count: *mut usize) -> MDStatus;
    pub fn md_result_plate(h: MDResultHandle, i: usize, plate: *mut *const c_char, color: *mut *const c_char) -> MDStatus;
    pub fn md_result_lpr_keypoints(h: MDResultHandle, i: usize, kps: *mut *const MDPoint, n: *mut usize) -> MDStatus;
    pub fn md_result_attribute(h: MDResultHandle, items: *mut *const MDAttrItem, count: *mut usize) -> MDStatus;
    pub fn md_result_attr_scores(h: MDResultHandle, i: usize, scores: *mut *const c_float, n: *mut usize) -> MDStatus;
    pub fn md_result_age(h: MDResultHandle, age: *mut c_int) -> MDStatus;
    pub fn md_result_age_batch(h: MDResultHandle, items: *mut *const c_int, count: *mut usize) -> MDStatus;
    pub fn md_result_gender(h: MDResultHandle, gender: *mut c_int) -> MDStatus;
    pub fn md_result_gender_batch(h: MDResultHandle, items: *mut *const c_int, count: *mut usize) -> MDStatus;
    pub fn md_result_ocr_batch_count(h: MDResultHandle, n: *mut usize) -> MDStatus;
    // FormulaRecognizer：单块 LaTeX（借用指针归结果句柄所有，立即复制）
    pub fn md_result_formula(h: MDResultHandle, i: usize, latex: *mut *const c_char) -> MDStatus;

    // ── 2D 批量结果 getter（按图索引 img，逐图取项数组） ──
    pub fn md_result_detection_batch(h: MDResultHandle, img: usize, items: *mut *const MDDetectionItem, count: *mut usize) -> MDStatus;
    pub fn md_result_classification_batch(h: MDResultHandle, img: usize, items: *mut *const MDClassifyItem, count: *mut usize) -> MDStatus;
    pub fn md_result_pose_batch(h: MDResultHandle, img: usize, items: *mut *const MDPoseItem, count: *mut usize) -> MDStatus;
    pub fn md_result_keypoints_batch(h: MDResultHandle, img: usize, item: usize, kps: *mut *const MDPoint3, n: *mut usize) -> MDStatus;
    pub fn md_result_obb_batch(h: MDResultHandle, img: usize, items: *mut *const MDObbItem, count: *mut usize) -> MDStatus;
    pub fn md_result_instance_seg_batch(h: MDResultHandle, img: usize, items: *mut *const MDIsegItem, count: *mut usize) -> MDStatus;
    pub fn md_result_mask_batch(h: MDResultHandle, img: usize, item: usize, buf: *mut *const u8, out_h: *mut usize, out_w: *mut usize) -> MDStatus;
    pub fn md_result_face_batch(h: MDResultHandle, img: usize, items: *mut *const MDFaceItem, count: *mut usize) -> MDStatus;
    pub fn md_result_face_kps_batch(h: MDResultHandle, img: usize, item: usize, kps: *mut *const MDPoint, n: *mut usize) -> MDStatus;
    pub fn md_result_face_embedding_batch(h: MDResultHandle, img: usize, emb: *mut *const c_float, n: *mut usize) -> MDStatus;
    pub fn md_result_reid_embedding_batch(h: MDResultHandle, img: usize, emb: *mut *const c_float, n: *mut usize) -> MDStatus;
    pub fn md_result_insightface_batch(h: MDResultHandle, img: usize, items: *mut *const MDInsightFaceItem, count: *mut usize) -> MDStatus;
    pub fn md_result_insightface_kps_batch(h: MDResultHandle, img: usize, item: usize, kps: *mut *const MDPoint, n: *mut usize) -> MDStatus;
    pub fn md_result_insightface_embedding_batch(h: MDResultHandle, img: usize, item: usize, emb: *mut *const c_float, n: *mut usize) -> MDStatus;
    pub fn md_result_insightface_pose_batch(h: MDResultHandle, img: usize, item: usize, pose: *mut *const c_float, n: *mut usize) -> MDStatus;
    pub fn md_result_ocr_batch(h: MDResultHandle, img: usize, line: usize, quad: *mut *const c_int, text: *mut *const c_char, score: *mut c_float) -> MDStatus;
    pub fn md_result_ocr_cls_batch(h: MDResultHandle, img: usize, line: usize, cls_label: *mut c_int, cls_score: *mut c_float) -> MDStatus;
    pub fn md_result_lpr_batch(h: MDResultHandle, img: usize, items: *mut *const MDLprItem, count: *mut usize) -> MDStatus;
    pub fn md_result_plate_batch(h: MDResultHandle, img: usize, item: usize, plate: *mut *const c_char, color: *mut *const c_char) -> MDStatus;
    pub fn md_result_lpr_keypoints_batch(h: MDResultHandle, img: usize, item: usize, kps: *mut *const MDPoint, n: *mut usize) -> MDStatus;
    pub fn md_result_attribute_batch(h: MDResultHandle, img: usize, items: *mut *const MDAttrItem, count: *mut usize) -> MDStatus;
    pub fn md_result_attr_scores_batch(h: MDResultHandle, img: usize, item: usize, scores: *mut *const c_float, n: *mut usize) -> MDStatus;

    // ── 绘制（基础） ──
    pub fn md_draw_rect(h: MDImageHandle, x: c_float, y: c_float, w: c_float, h: c_float,
        color: MDColorRGBA, alpha: c_float) -> MDStatus;
    pub fn md_draw_polygon(h: MDImageHandle, xs: *const c_float, ys: *const c_float, n: usize,
        color: MDColorRGBA, alpha: c_float) -> MDStatus;
    pub fn md_draw_text(h: MDImageHandle, x: c_float, y: c_float, text: *const c_char,
        font_path: *const c_char, font_size: c_int, color: MDColorRGBA, alpha: c_float) -> MDStatus;

    // ── 结果可视化（句柄直达 C++ vis_*） ──
    pub fn md_draw_result(img: MDImageHandle, result: MDResultHandle, opt: *const MDDrawOptions) -> MDStatus;

    // ── 多目标跟踪器 ──
    pub fn md_tracker_create(kind: c_int, out: *mut MDTrackerHandle) -> MDStatus;
    pub fn md_tracker_destroy(h: MDTrackerHandle);
    pub fn md_tracker_capacity(h: MDTrackerHandle, boxes: *const MDBox,
        scores: *const c_float, label_ids: *const c_int, n: usize,
        out_count: *mut usize) -> MDStatus;
    pub fn md_tracker_update(h: MDTrackerHandle, boxes: *const MDBox,
        scores: *const c_float, label_ids: *const c_int, n: usize,
        out: *mut MDTrackItem, out_count: *mut usize) -> MDStatus;
    pub fn md_tracker_set_params(h: MDTrackerHandle, name: *const c_char, value: f64) -> MDStatus;
    pub fn md_tracker_reset(h: MDTrackerHandle) -> MDStatus;

    // ── 条码 / 二维码 ──
    pub fn md_barcode_create(out: *mut MDBarcodeHandle) -> MDStatus;
    pub fn md_barcode_destroy(h: MDBarcodeHandle);
    pub fn md_barcode_set_formats(h: MDBarcodeHandle, formats: c_uint) -> MDStatus;
    pub fn md_barcode_detect(h: MDBarcodeHandle, img: MDImageHandle,
        items: *mut MDBarcodeItem, count: *mut c_uint) -> MDStatus;
}
