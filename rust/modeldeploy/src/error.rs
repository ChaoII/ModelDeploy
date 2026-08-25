use crate::ffi::MDStatus;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum MdError {
    #[error("空指针")]
    NullPointer,

    #[error("参数不合法: {0}")]
    InvalidArgument(String),

    #[error("参数类型不匹配: {0}")]
    InvalidType(String),

    #[error("路径不存在: {0}")]
    PathNotFound(String),

    #[error("模型加载失败: {0}")]
    ModelLoad(String),

    #[error("模型推理失败: {0}")]
    ModelPredict(String),

    #[error("模型初始化失败: {0}")]
    ModelInit(String),

    #[error("模型类型不支持")]
    UnsupportedType,

    #[error("后端不可用")]
    UnsupportedBackend,

    #[error("设备不支持: {0}")]
    Unsupported(&'static str),

    #[error("内存分配失败")]
    OutOfMemory,

    #[error("图像解码失败")]
    ImageDecode,

    #[error("句柄并发使用")]
    Busy,

    #[error("音频解码失败")]
    AudioDecode,

    #[error("功能未实现")]
    NotImplemented,

    #[error("未知错误 (code={0})")]
    Unknown(i32),
}

impl MdError {
    /// 从 C API 返回值 + 最近错误信息构造
    pub fn from_status(code: MDStatus, last_error: &str) -> Self {
        let msg = |what: &str| {
            if last_error.is_empty() {
                what.to_string()
            } else {
                format!("{}: {}", what, last_error)
            }
        };
        match code {
            MDStatus::OK => unreachable!("OK is not an error"),
            MDStatus::ERR_NULL_POINTER => MdError::NullPointer,
            MDStatus::ERR_INVALID_ARGUMENT => MdError::InvalidArgument(msg("invalid argument")),
            MDStatus::ERR_INVALID_TYPE => MdError::InvalidType(msg("invalid type")),
            MDStatus::ERR_PATH_NOT_FOUND => MdError::PathNotFound(msg("path not found")),
            MDStatus::ERR_MODEL_LOAD => MdError::ModelLoad(msg("model load")),
            MDStatus::ERR_MODEL_PREDICT => MdError::ModelPredict(msg("predict")),
            MDStatus::ERR_MODEL_INIT => MdError::ModelInit(msg("model init")),
            MDStatus::ERR_UNSUPPORTED_TYPE => MdError::UnsupportedType,
            MDStatus::ERR_UNSUPPORTED_BACKEND => MdError::UnsupportedBackend,
            MDStatus::ERR_OUT_OF_MEMORY => MdError::OutOfMemory,
            MDStatus::ERR_IMAGE_DECODE => MdError::ImageDecode,
            MDStatus::ERR_BUSY => MdError::Busy,
            MDStatus::ERR_AUDIO_DECODE => MdError::AudioDecode,
            MDStatus::ERR_NOT_IMPLEMENTED => MdError::NotImplemented,
        }
    }
}

/// 检查 C API 返回值，非 OK 则返回 Err（附最近错误信息）
pub fn check_status(code: MDStatus) -> Result<(), MdError> {
    if code == MDStatus::OK {
        Ok(())
    } else {
        let err = unsafe { crate::ffi::md_get_last_error() };
        let last_error = if err.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned() }
        };
        Err(MdError::from_status(code, &last_error))
    }
}
