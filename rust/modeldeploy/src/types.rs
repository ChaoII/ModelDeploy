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

/// 行人重识别（ReID）结果（512-d embedding）
#[derive(Debug, Clone)]
pub struct ReIdResult {
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

/// 模型类型（对应 capi MDModelKind）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Detection,
    Classification,
    Pose,
    Hand,
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
    ReId,
    SpeakerVerify,
}

impl ModelKind {
    pub(crate) fn to_ffi(self) -> ffi::MDModelKind {
        use ffi::MDModelKind::*;
        match self {
            ModelKind::Detection => DETECTION,
            ModelKind::Classification => CLASSIFICATION,
            ModelKind::Pose => POSE,
            ModelKind::Hand => HAND,
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
            ModelKind::ReId => REID,
            ModelKind::SpeakerVerify => SPEAKER_VERIFY,
        }
    }
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// 图像格式（对应 C++ ImageType 枚举值）
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// 变体名故意与 C 枚举一致（GRAY_U8/PLA_BGR_U8/... 等含下划线），不宜改为 camelCase。
#[allow(non_camel_case_types)]
pub enum ImageFormat {
    GRAY_U8 = 0,
    PLA_BGR_U8 = 20,
    PLA_RGB_U8 = 21,
    PKG_BGR_U8 = 22,
    PKG_RGB_U8 = 23,
    PLA_BGRA_U8 = 24,
    PLA_RGBA_U8 = 25,
    PKG_BGRA_U8 = 26,
    PKG_RGBA_U8 = 27,
    NV12 = 60,
    NV21 = 61,
    I420 = 62,
    Unknown = 63,
}

impl From<i32> for ImageFormat {
    fn from(v: i32) -> Self {
        match v {
            0 => ImageFormat::GRAY_U8,
            20 => ImageFormat::PLA_BGR_U8,
            21 => ImageFormat::PLA_RGB_U8,
            22 => ImageFormat::PKG_BGR_U8,
            23 => ImageFormat::PKG_RGB_U8,
            24 => ImageFormat::PLA_BGRA_U8,
            25 => ImageFormat::PLA_RGBA_U8,
            26 => ImageFormat::PKG_BGRA_U8,
            27 => ImageFormat::PKG_RGBA_U8,
            60 => ImageFormat::NV12,
            61 => ImageFormat::NV21,
            62 => ImageFormat::I420,
            _ => ImageFormat::Unknown,
        }
    }
}

/// 跟踪器类型（对应 capi MDTrackerKind）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackerKind {
    ByteTrack,
    BotSort,
    StrongSort,
}

impl TrackerKind {
    pub(crate) fn to_ffi(self) -> i32 {
        match self {
            TrackerKind::ByteTrack => 0,
            TrackerKind::BotSort => 1,
            TrackerKind::StrongSort => 2,
        }
    }
}

impl TryFrom<i32> for TrackerKind {
    type Error = ();
    fn try_from(v: i32) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(TrackerKind::ByteTrack),
            1 => Ok(TrackerKind::BotSort),
            2 => Ok(TrackerKind::StrongSort),
            _ => Err(()),
        }
    }
}

/// 跟踪目标状态（对应 capi MDTrackState）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3,
}

impl TryFrom<i32> for TrackState {
    type Error = ();
    fn try_from(v: i32) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(TrackState::New),
            1 => Ok(TrackState::Tracked),
            2 => Ok(TrackState::Lost),
            3 => Ok(TrackState::Removed),
            _ => Err(()),
        }
    }
}

/// 跟踪目标（跨帧 ID 稳定）
#[derive(Debug, Clone)]
pub struct TrackItem {
    pub rect: Rect,
    pub track_id: i32,
    pub label_id: i32,
    pub score: f32,
    pub state: TrackState,
}

/// 图像平面（原始指针 + 行步长），零拷贝访问，不拥有内存
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub data: *const u8,
    pub step: i32,
}

/// 条码 / 二维码格式（对应 capi format 字符串，如 "QR Code"）
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BarcodeFormat {
    QrCode,
    DataMatrix,
    Aztec,
    Ean8,
    Ean13,
    Upca,
    Upce,
    Code128,
    Code39,
    Code93,
    Itf,
    Codabar,
    Other(String),
}

impl BarcodeFormat {
    /// 由 capi 传入的格式名（ZXing::ToString，如 "QR Code" / "EAN-13"）解析。
    pub fn from_name(s: &str) -> Self {
        match s.to_ascii_uppercase().replace('_', " ").trim() {
            "QR CODE" => BarcodeFormat::QrCode,
            "DATA MATRIX" => BarcodeFormat::DataMatrix,
            "AZTEC" => BarcodeFormat::Aztec,
            "EAN-8" => BarcodeFormat::Ean8,
            "EAN-13" => BarcodeFormat::Ean13,
            "UPCA" | "UPC-A" => BarcodeFormat::Upca,
            "UPCE" | "UPC-E" => BarcodeFormat::Upce,
            "CODE 128" => BarcodeFormat::Code128,
            "CODE 39" => BarcodeFormat::Code39,
            "CODE 93" => BarcodeFormat::Code93,
            "ITF" | "INTERLEAVED 2 OF 5" => BarcodeFormat::Itf,
            "CODABAR" => BarcodeFormat::Codabar,
            other => BarcodeFormat::Other(other.to_string()),
        }
    }
}

/// 条码 / 二维码解码结果
#[derive(Debug, Clone)]
pub struct BarcodeResult {
    pub text: String,
    pub format: BarcodeFormat,
    pub quad: [(f32, f32); 4],
    pub score: f32,
    pub is_qr: bool,
}
