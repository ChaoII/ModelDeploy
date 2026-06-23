use crate::ffi::*;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum MdError {
    #[error("模型文件路径不存在: {0}")]
    PathNotFound(String),

    #[error("模型初始化失败: {0}")]
    ModelInitFailed(String),

    #[error("模型预测失败: {0}")]
    PredictFailed(String),

    #[error("内存分配失败")]
    OutOfMemory,

    #[error("模型类型不匹配: 期望 {expected:?}, 实际 {actual:?}")]
    TypeMismatch { expected: MDModelType, actual: MDModelType },

    #[error("文件打开失败: {0}")]
    FileOpenFailed(String),

    #[error("图像文件无法读取: {0}")]
    ImageReadFailed(String),

    #[error("后端调用错误")]
    CallError,

    #[error("写入 WAV 文件失败")]
    WriteWaveFailed,

    #[error("未知错误 (code={0})")]
    Unknown(i32),
}

impl From<MDStatusCode> for MdError {
    fn from(code: MDStatusCode) -> Self {
        if code.0 == MDStatusCode_Success.0 {
            unreachable!()
        }
        match code {
            MDStatusCode_PathNotFound => MdError::PathNotFound("".into()),
            MDStatusCode_FileOpenFailed => MdError::FileOpenFailed("".into()),
            MDStatusCode_CallError => MdError::CallError,
            MDStatusCode_ModelInitializeFailed => MdError::ModelInitFailed("".into()),
            MDStatusCode_ModelPredictFailed => MdError::PredictFailed("".into()),
            MDStatusCode_MemoryAllocatedFailed => MdError::OutOfMemory,
            MDStatusCode_ModelTypeError => MdError::TypeMismatch {
                expected: MDModelType_Classification,
                actual: MDModelType_Classification,
            },
            MDStatusCode_WriteWaveFailed => MdError::WriteWaveFailed,
            _ => MdError::Unknown(code.0),
        }
    }
}

/// 检查 C API 返回值，非 Success 则返回 Err
pub fn check_status(code: MDStatusCode) -> Result<(), MdError> {
    if code.0 == MDStatusCode_Success.0 {
        Ok(())
    } else {
        Err(MdError::from(code))
    }
}
