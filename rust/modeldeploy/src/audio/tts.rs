use crate::error::{check_status, MdError};
use crate::ffi;
use crate::runtime::RuntimeOption;
use std::ffi::CString;
use std::ptr;

/// Kokoro TTS 模型封装
#[derive(Debug)]
pub struct Kokoro {
    model: ffi::MDModel,
}

unsafe impl Send for Kokoro {}
unsafe impl Sync for Kokoro {}

impl Kokoro {
    /// 加载 TTS 模型
    pub fn new(
        model_path: &str,
        tokens_path: &str,
        lexicons_path: &str,
        voice_path: &str,
        jieba_path: &str,
        option: &RuntimeOption,
    ) -> Result<Self, MdError> {
        // 注意：MDKokoroParameters 的字段是 *mut c_char，需要确保生命周期
        let c_model = CString::new(model_path).unwrap();
        let c_tokens = CString::new(tokens_path).unwrap();
        let c_lexicons = CString::new(lexicons_path).unwrap();
        let c_voice = CString::new(voice_path).unwrap();
        let c_jieba = CString::new(jieba_path).unwrap();

        let params = ffi::MDKokoroParameters {
            model: c_model.as_ptr() as *mut i8,
            tokens: c_tokens.as_ptr() as *mut i8,
            lexicons: c_lexicons.as_ptr() as *mut i8,
            voice: c_voice.as_ptr() as *mut i8,
            jieba: c_jieba.as_ptr() as *mut i8,
        };

        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(),
            type_: ffi::MDModelType_TTS,
            format: ffi::MDModelFormat_ONNX,
            model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_kokoro_model(&mut model, &params, &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() {
            return Err(MdError::ModelInitFailed(model_path.into()));
        }
        Ok(Self { model })
    }

    /// 文本转语音
    pub fn predict(&self, text: &str) -> Result<TtsResult, MdError> {
        let ctext = CString::new(text).map_err(|_| MdError::CallError)?;
        let mut result = ffi::MDTTSResult {
            data: ptr::null_mut(),
            size: 0,
            sample_rate: 0,
        };
        let status = unsafe { ffi::md_kokoro_model_predict(&self.model, ctext.as_ptr(), &mut result) };
        check_status(status)?;

        let audio = if result.size > 0 && !result.data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(result.data, result.size as usize) };
            slice.to_vec()
        } else {
            Vec::new()
        };

        let sample_rate = result.sample_rate;
        unsafe { ffi::md_free_kokoro_result(&mut result) };

        Ok(TtsResult {
            audio,
            sample_rate,
        })
    }

    pub fn is_initialized(&self) -> bool {
        !self.model.model_content.is_null()
    }
}

impl Drop for Kokoro {
    fn drop(&mut self) {
        if !self.model.model_content.is_null() {
            unsafe { ffi::md_free_kokoro_model(&mut self.model) };
        }
    }
}

/// TTS 推理结果
#[derive(Debug, Clone)]
pub struct TtsResult {
    /// PCM float32 音频数据
    pub audio: Vec<f32>,
    /// 采样率（Hz）
    pub sample_rate: i32,
}
