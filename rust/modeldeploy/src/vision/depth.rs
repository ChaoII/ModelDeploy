use crate::error::{check_status, MdError};
use crate::ffi;
use crate::image::Image;
use crate::runtime::RuntimeOption;
use std::ffi::CString;
use std::ptr;

/// 深度估计结果（yolo26n-depth，log 深度经 exp 还原为米）
/// `depth` 为逐像素深度值（米），长度 = shape[0] * shape[1]
#[derive(Debug, Clone)]
pub struct DepthResult {
    pub depth: Vec<f32>,
    pub shape: Vec<i32>,
}

/// Ultralytics 深度估计模型
#[derive(Debug)]
pub struct UltralyticsDepth {
    model: ffi::MDModel,
}
unsafe impl Send for UltralyticsDepth {}
unsafe impl Sync for UltralyticsDepth {}

impl UltralyticsDepth {
    /// 加载深度估计模型
    pub fn new(path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::PathNotFound(path.into()))?;
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(),
            type_: ffi::MDModelType_Depth,
            format: ffi::MDModelFormat_ONNX,
            model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_depth_model(&mut model, cpath.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() {
            return Err(MdError::ModelInitFailed(path.into()));
        }
        Ok(Self { model })
    }

    /// 设置输入尺寸
    pub fn set_input_size(&mut self, width: i32, height: i32) -> Result<(), MdError> {
        let size = ffi::MDSize { width, height };
        let status = unsafe { ffi::md_set_depth_input_size(&self.model, size) };
        check_status(status)
    }

    /// 推理
    pub fn predict(&self, image: &Image) -> Result<DepthResult, MdError> {
        let mut raw = ffi::MDDepthResult {
            depth: ptr::null_mut(),
            shape: ptr::null_mut(),
            shape_size: 0,
        };
        let status = unsafe { ffi::md_depth_predict(&self.model, &image.raw, &mut raw) };
        check_status(status)?;

        // C 层分配的 shape 为 `new int[]`，先拷贝出来以便计算 depth 长度
        let shape = if raw.shape_size > 0 && !raw.shape.is_null() {
            unsafe { std::slice::from_raw_parts(raw.shape, raw.shape_size as usize) }.to_vec()
        } else {
            Vec::new()
        };
        // depth 长度等于 shape 各维度乘积（H*W）
        let num_pixels = if shape.is_empty() {
            0
        } else {
            shape.iter().fold(1usize, |acc, &d| acc.saturating_mul(d.max(0) as usize))
        };
        let depth = if num_pixels > 0 && !raw.depth.is_null() {
            unsafe { std::slice::from_raw_parts(raw.depth, num_pixels) }.to_vec()
        } else {
            Vec::new()
        };

        // 释放 C 层分配的 depth/shape 内存（new[]/delete[] 配对）
        unsafe { ffi::md_free_depth_result(&mut raw) };
        Ok(DepthResult { depth, shape })
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

impl Drop for UltralyticsDepth {
    fn drop(&mut self) {
        if !self.model.model_content.is_null() {
            unsafe { ffi::md_free_depth_model(&mut self.model) };
        }
    }
}
