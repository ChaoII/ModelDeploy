use crate::ffi;
use std::fmt;

/// 浮点矩形
#[derive(Debug, Clone, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// 2D 点
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

/// 3D 关键点
#[derive(Debug, Clone, PartialEq)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 旋转框（OBB）
#[derive(Debug, Clone, PartialEq)]
pub struct RotatedBox {
    pub cx: f32,
    pub cy: f32,
    pub width: f32,
    pub height: f32,
    pub angle: f32,
}

/// 检测结果
#[derive(Debug, Clone)]
pub struct Detection {
    pub rect: Rect,
    pub label_id: i32,
    pub score: f32,
}

/// 分类结果
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub label_id: i32,
    pub score: f32,
}

/// 姿态结果（骨架）
#[derive(Debug, Clone)]
pub struct Pose {
    pub rect: Rect,
    pub score: f32,
    pub keypoints: Vec<Point3>,
}

/// OBB 旋转框结果
#[derive(Debug, Clone)]
pub struct Obb {
    pub rotated_box: RotatedBox,
    pub label_id: i32,
    pub score: f32,
}

/// 实例分割结果
#[derive(Debug, Clone)]
pub struct InstanceSeg {
    pub rect: Rect,
    pub label_id: i32,
    pub score: f32,
}

/// 语义分割结果（整图 label）
#[derive(Debug, Clone)]
pub struct SemSeg {
    pub labels: Vec<u8>,
    pub height: usize,
    pub width: usize,
    pub num_classes: i32,
}

/// 深度估计结果
#[derive(Debug, Clone)]
pub struct Depth {
    pub depth: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

/// 人脸检测结果
#[derive(Debug, Clone)]
pub struct FaceDetection {
    pub rect: Rect,
    pub score: f32,
    pub keypoints: Vec<Point>,
}

/// 人脸识别结果（embedding）
#[derive(Debug, Clone)]
pub struct FaceRecognition {
    pub embedding: Vec<f32>,
}

/// InsightFace 完整分析结果
#[derive(Debug, Clone)]
pub struct InsightFace {
    pub rect: Rect,
    pub score: f32,
    pub keypoints: Vec<Point>,
    pub embedding: Vec<f32>,
    pub pose: Vec<f32>,
    pub gender: i32,
    pub age: i32,
}

/// OCR 结果
#[derive(Debug, Clone)]
pub struct OcrLine {
    pub quad: [i32; 8],
    pub text: String,
    pub score: f32,
    pub cls_label: i32,
    pub cls_score: f32,
}

/// 车牌结果
#[derive(Debug, Clone)]
pub struct LicensePlate {
    pub rect: Rect,
    pub plate: String,
    pub color: String,
    pub score: f32,
    pub keypoints: Vec<Point>,
}

/// 行人属性结果
#[derive(Debug, Clone)]
pub struct Attribute {
    pub rect: Rect,
    pub box_label_id: i32,
    pub box_score: f32,
    pub attr_scores: Vec<f32>,
}

/// ASR 识别结果
#[derive(Debug, Clone)]
pub struct AsrText {
    pub text: String,
}

/// TTS 合成结果
#[derive(Debug, Clone)]
pub struct TtsAudio {
    pub samples: Vec<f32>,
    pub sample_rate: i32,
}

/// 模型类型（对应 capi2 MDModelKind）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Detection,
    Classification,
    Pose,
    Obb,
    InstanceSeg,
    SemSeg,
    Depth,
    FaceDet,
    FaceRec,
    FaceAge,
    FaceGender,
    FaceAs,
    FaceRecPipeline,
    InsightFace,
    InsightFaceDet,
    Ocr,
    OcrDet,
    OcrRec,
    OcrCls,
    LprDet,
    LprRec,
    LprPipeline,
    PedestrianAttribute,
    Asr,
    Tts,
}

impl ModelKind {
    pub(crate) fn to_ffi(self) -> ffi::MDModelKind {
        use ffi::MDModelKind::*;
        match self {
            ModelKind::Detection => DETECTION,
            ModelKind::Classification => CLASSIFICATION,
            ModelKind::Pose => POSE,
            ModelKind::Obb => OBB,
            ModelKind::InstanceSeg => INSTANCE_SEG,
            ModelKind::SemSeg => SEM_SEG,
            ModelKind::Depth => DEPTH,
            ModelKind::FaceDet => FACE_DET,
            ModelKind::FaceRec => FACE_REC,
            ModelKind::FaceAge => FACE_AGE,
            ModelKind::FaceGender => FACE_GENDER,
            ModelKind::FaceAs => FACE_AS,
            ModelKind::FaceRecPipeline => FACE_REC_PIPELINE,
            ModelKind::InsightFace => INSIGHTFACE,
            ModelKind::InsightFaceDet => INSIGHTFACE_DET,
            ModelKind::Ocr => OCR,
            ModelKind::OcrDet => OCR_DET,
            ModelKind::OcrRec => OCR_REC,
            ModelKind::OcrCls => OCR_CLS,
            ModelKind::LprDet => LPR_DET,
            ModelKind::LprRec => LPR_REC,
            ModelKind::LprPipeline => LPR_PIPELINE,
            ModelKind::PedestrianAttribute => PED_ATTR,
            ModelKind::Asr => ASR,
            ModelKind::Tts => TTS,
        }
    }
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
