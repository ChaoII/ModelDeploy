use crate::error::{check_status, MdError};
use crate::ffi;
use crate::image::Image;
use crate::runtime::RuntimeOption;
use std::ffi::CString;
use std::ptr;

/// 语义分割结果（yolo26n-sem，cityscapes 19 类）
/// `labels` 为逐像素类别索引（[0, num_classes)），长度 = shape[0] * shape[1]
#[derive(Debug, Clone)]
pub struct SemSegResult {
    pub labels: Vec<u8>,
    pub shape: Vec<i32>,
    pub num_classes: i32,
}

/// Ultralytics 语义分割模型
#[derive(Debug)]
pub struct UltralyticsSem {
    model: ffi::MDModel,
}
unsafe impl Send for UltralyticsSem {}
unsafe impl Sync for UltralyticsSem {}

impl UltralyticsSem {
    /// 加载语义分割模型
    pub fn new(path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::PathNotFound(path.into()))?;
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(),
            type_: ffi::MDModelType_SemSeg,
            format: ffi::MDModelFormat_ONNX,
            model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_sem_model(&mut model, cpath.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() {
            return Err(MdError::ModelInitFailed(path.into()));
        }
        Ok(Self { model })
    }

    /// 设置输入尺寸
    pub fn set_input_size(&mut self, width: i32, height: i32) -> Result<(), MdError> {
        let size = ffi::MDSize { width, height };
        let status = unsafe { ffi::md_set_sem_input_size(&self.model, size) };
        check_status(status)
    }

    /// 推理
    pub fn predict(&self, image: &Image) -> Result<SemSegResult, MdError> {
        let mut raw = ffi::MDSemSegResult {
            labels: ptr::null_mut(),
            shape: ptr::null_mut(),
            shape_size: 0,
            num_classes: 0,
        };
        let status = unsafe { ffi::md_sem_predict(&self.model, &image.raw, &mut raw) };
        check_status(status)?;

        // C 层分配的 shape 为 `new int[]`，先拷贝出来以便计算 labels 长度
        let shape = if raw.shape_size > 0 && !raw.shape.is_null() {
            unsafe { std::slice::from_raw_parts(raw.shape, raw.shape_size as usize) }.to_vec()
        } else {
            Vec::new()
        };
        // labels 长度等于 shape 各维度乘积（H*W）
        let num_pixels = if shape.is_empty() {
            0
        } else {
            shape.iter().fold(1usize, |acc, &d| acc.saturating_mul(d.max(0) as usize))
        };
        let labels = if num_pixels > 0 && !raw.labels.is_null() {
            unsafe { std::slice::from_raw_parts(raw.labels, num_pixels) }.to_vec()
        } else {
            Vec::new()
        };
        let num_classes = raw.num_classes;

        // 释放 C 层分配的 labels/shape 内存（new[]/delete[] 配对）
        unsafe { ffi::md_free_sem_result(&mut raw) };
        Ok(SemSegResult { labels, shape, num_classes })
    }

    /// 克隆模型，创建独立的推理副本
    pub fn try_clone(&self) -> Result<Self, MdError> {
        let mut raw = ffi::MDModel {
            model_name: std::ptr::null_mut(),
            type_: self.model.type_,
            format: self.model.format,
            model_content: std::ptr::null_mut(),
        };
        let status = unsafe { ffi::md_clone_model(&mut raw, &self.model) };
        check_status(status)?;
        Ok(Self { model: raw })
    }

    /// 模型是否已初始化
    pub fn is_initialized(&self) -> bool {
        !self.model.model_content.is_null()
    }
}

impl Drop for UltralyticsSem {
    fn drop(&mut self) {
        if !self.model.model_content.is_null() {
            unsafe { ffi::md_free_sem_model(&mut self.model) };
        }
    }
}
