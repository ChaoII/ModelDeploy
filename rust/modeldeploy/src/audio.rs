use crate::error::{check_status, MdError};
use crate::ffi;
use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;

/// 说话人检索（capi md_audio_speaker_search_*）。
pub struct SpeakerSearch {
    handle: ffi::MDAudioSolutionHandle,
}

unsafe impl Send for SpeakerSearch {}
unsafe impl Sync for SpeakerSearch {}

impl SpeakerSearch {
    pub fn new() -> Result<Self, MdError> {
        let mut h = ptr::null_mut();
        check_status(unsafe {
            ffi::md_audio_solution_create(&mut h, ffi::MDAudioSolutionKind::SpeakerSearch as i32)
        })?;
        Ok(Self { handle: h })
    }

    pub fn enroll(&self, label: &str, emb: &[f32]) -> Result<(), MdError> {
        let c = CString::new(label).map_err(|_| MdError::InvalidArgument("label".into()))?;
        check_status(unsafe {
            ffi::md_audio_speaker_search_enroll(self.handle, c.as_ptr(), emb.as_ptr(), emb.len())
        })
    }

    pub fn match_top(&self, emb: &[f32]) -> Result<String, MdError> {
        let mut label: *const libc::c_char = ptr::null();
        let mut score = 0.0f32;
        check_status(unsafe {
            ffi::md_audio_speaker_search_match(self.handle, emb.as_ptr(), emb.len(), 1, &mut label, &mut score)
        })?;
        if label.is_null() {
            return Err(MdError::ModelPredict("no match".into()));
        }
        Ok(unsafe { CStr::from_ptr(label) }.to_string_lossy().into_owned())
    }
}

impl Drop for SpeakerSearch {
    fn drop(&mut self) {
        unsafe { ffi::md_audio_solution_destroy(self.handle) };
    }
}

/// 音频工具：重采样。
pub fn resample(input: &[f32], in_sr: i32, out_sr: i32) -> Result<Vec<f32>, MdError> {
    let mut out: *const f32 = ptr::null();
    let mut n = 0usize;
    check_status(unsafe { ffi::md_audio_resample(input.as_ptr(), input.len(), in_sr, out_sr, &mut out, &mut n) })?;
    if out.is_null() {
        return Ok(Vec::new());
    }
    Ok(unsafe { std::slice::from_raw_parts(out, n) }.to_vec())
}
