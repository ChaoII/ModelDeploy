use crate::error::{check_status, MdError};
use crate::ffi;
use crate::model::Model;
use crate::runtime::RuntimeOption;
use crate::types::ModelKind;
use std::ffi::CStr;
use std::ffi::CString;

/// NLP 工具：按标点分句。
pub fn split_sentences(text: &str) -> Result<Vec<String>, MdError> {
    let c = CString::new(text).map_err(|_| MdError::InvalidArgument("text".into()))?;
    let mut sents: *const *const libc::c_char = std::ptr::null();
    let mut n = 0usize;
    check_status(unsafe { ffi::md_nlp_split_sent(c.as_ptr(), &mut sents, &mut n) })?;
    let mut out = Vec::with_capacity(n);
    if !sents.is_null() {
        for i in 0..n {
            let p = unsafe { *sents.add(i) };
            if !p.is_null() {
                out.push(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned());
            }
        }
    }
    Ok(out)
}

/// NLP 工具：文本统计（字符数 / 词数 / 句数）。
pub fn stats(text: &str) -> Result<(usize, usize, usize), MdError> {
    let c = CString::new(text).map_err(|_| MdError::InvalidArgument("text".into()))?;
    let (mut chars, mut words, mut sents) = (0usize, 0usize, 0usize);
    check_status(unsafe { ffi::md_nlp_stats(c.as_ptr(), &mut chars, &mut words, &mut sents) })?;
    Ok((chars, words, sents))
}

/// 文本分类器（capi MD_MODEL_TEXT_CLASSIFIER / md_nlp_classify）。
pub struct NlpClassifier {
    model: Model,
}

impl NlpClassifier {
    pub fn new(model_path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        Ok(Self { model: Model::new(ModelKind::TextClassifier, model_path, option)? })
    }

    pub fn predict(&self, text: &str) -> Result<(i32, f32), MdError> {
        let c = CString::new(text).map_err(|_| MdError::InvalidArgument("text".into()))?;
        let mut label = 0i32;
        let mut score = 0.0f32;
        check_status(unsafe { ffi::md_nlp_classify(self.model.handle, c.as_ptr(), &mut label, &mut score) })?;
        Ok((label, score))
    }
}
