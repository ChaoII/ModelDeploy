use crate::error::{check_status, MdError};
use crate::ffi;
use crate::image::Image;
use crate::runtime::RuntimeOption;
use crate::types::Classification;
use std::ffi::CString;
use std::ptr;

/// Ultralytics 分类模型封装
#[derive(Debug)]
pub struct UltralyticsCls {
    model: ffi::MDModel,
}

unsafe impl Send for UltralyticsCls {}
unsafe impl Sync for UltralyticsCls {}

impl UltralyticsCls {
    /// 加载分类模型
    pub fn new(path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::PathNotFound(path.into()))?;
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(),
            type_: ffi::MDModelType_Classification,
            format: ffi::MDModelFormat_ONNX,
            model_content: ptr::null_mut(),
        };
        let status = unsafe {
            crate::ffi::md_create_classification_model(&mut model, cpath.as_ptr(), &option.raw)
        };
        check_status(status)?;
        if model.model_content.is_null() {
            return Err(MdError::ModelInitFailed(path.into()));
        }
        Ok(Self { model })
    }

    /// 推理
    pub fn predict(&self, image: &Image, topk: i32) -> Result<Vec<Classification>, MdError> {
        let mut results = ffi::MDClassificationResults {
            data: ptr::null_mut(),
            size: 0,
        };
        let status = unsafe {
            crate::ffi::md_classification_predict(
                &self.model,
                &image.raw,
                topk,
                &mut results,
            )
        };
        check_status(status)?;

        let classifications = if results.size > 0 && !results.data.is_null() {
            let slice =
                unsafe { std::slice::from_raw_parts(results.data, results.size as usize) };
            slice
                .iter()
                .map(|r| Classification {
                    label_id: r.label_id,
                    score: r.score,
                    label_name: String::new(),
                })
                .collect()
        } else {
            Vec::new()
        };

        unsafe { crate::ffi::md_free_classification_result(&mut results) };
        Ok(classifications)
    }

    /// 模型是否已初始化
    pub fn is_initialized(&self) -> bool {
        !self.model.model_content.is_null()
    }
}

impl Drop for UltralyticsCls {
    fn drop(&mut self) {
        if !self.model.model_content.is_null() {
            unsafe { crate::ffi::md_free_classification_model(&mut self.model) };
        }
    }
}
