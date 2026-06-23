use crate::ffi;
use std::ffi::CStr;
use std::fmt;

/// 检测框
#[derive(Debug, Clone, PartialEq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl From<ffi::MDRect> for Rect {
    fn from(r: ffi::MDRect) -> Self {
        Self {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
        }
    }
}

impl From<Rect> for ffi::MDRect {
    fn from(r: Rect) -> Self {
        Self {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
        }
    }
}

/// 2D 点
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl From<ffi::MDPoint> for Point {
    fn from(p: ffi::MDPoint) -> Self {
        Self { x: p.x, y: p.y }
    }
}

/// 3D 浮点关键点
#[derive(Debug, Clone, PartialEq)]
pub struct Point3f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<ffi::MDPoint3f> for Point3f {
    fn from(p: ffi::MDPoint3f) -> Self {
        Self { x: p.x, y: p.y, z: p.z }
    }
}

/// 尺寸
#[derive(Debug, Clone, PartialEq)]
pub struct Size {
    pub width: i32,
    pub height: i32,
}

/// 单个检测结果
#[derive(Debug, Clone)]
pub struct Detection {
    pub rect: Rect,
    pub label_id: i32,
    pub score: f32,
    pub label_name: String,
}

/// 分类结果
#[derive(Debug, Clone)]
pub struct Classification {
    pub label_id: i32,
    pub score: f32,
    pub label_name: String,
}

/// 人脸检测结果
#[derive(Debug, Clone)]
pub struct FaceDetection {
    pub rect: Rect,
    pub score: f32,
    pub landmarks: Vec<Point3f>,
}

/// 模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Classification,
    Detection,
    Keypoint,
    Ocr,
    Face,
    Lpr,
    Pipeline,
    Asr,
    Tts,
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelType::Classification => write!(f, "Classification"),
            ModelType::Detection => write!(f, "Detection"),
            ModelType::Keypoint => write!(f, "Keypoint"),
            ModelType::Ocr => write!(f, "OCR"),
            ModelType::Face => write!(f, "Face"),
            ModelType::Lpr => write!(f, "LPR"),
            ModelType::Pipeline => write!(f, "Pipeline"),
            ModelType::Asr => write!(f, "ASR"),
            ModelType::Tts => write!(f, "TTS"),
        }
    }
}

impl From<ffi::MDModelType> for ModelType {
    fn from(t: ffi::MDModelType) -> Self {
        match t.0 {
            0 => ModelType::Classification,
            1 => ModelType::Detection,
            2 => ModelType::Keypoint,
            3 => ModelType::Ocr,
            4 => ModelType::Face,
            5 => ModelType::Lpr,
            6 => ModelType::Pipeline,
            7 => ModelType::Asr,
            8 => ModelType::Tts,
            _ => ModelType::Detection,
        }
    }
}

/// 将 C 字符串指针转为 Rust String
#[allow(dead_code)]
pub(crate) unsafe fn c_str_to_string(s: *const libc::c_char) -> String {
    if s.is_null() {
        String::new()
    } else {
        CStr::from_ptr(s).to_string_lossy().into_owned()
    }
}

/// 将 Rust 字符串转为 C 字符串指针（使用时需确保生命周期）
#[allow(dead_code)]
pub(crate) fn string_to_c_str(s: &str) -> Vec<u8> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    bytes
}
