use crate::error::{check_status, MdError};
use crate::ffi;
use crate::image::Image;
use crate::runtime::RuntimeOption;
use crate::types::{Point3f, Rect};
use std::ffi::CString;
use std::ptr;

/// OBB 检测结果
#[derive(Debug, Clone)]
pub struct ObbResult {
    pub xc: f32, pub yc: f32, pub width: f32, pub height: f32, pub angle: f32,
    pub label_id: i32, pub score: f32,
}

/// Ultralytics OBB 模型
#[derive(Debug)]
pub struct UltralyticsObb {
    model: ffi::MDModel,
}
unsafe impl Send for UltralyticsObb {}
unsafe impl Sync for UltralyticsObb {}

impl UltralyticsObb {
    pub fn new(path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::PathNotFound(path.into()))?;
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(), type_: ffi::MDModelType_Detection,
            format: ffi::MDModelFormat_ONNX, model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_obb_model(&mut model, cpath.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() { return Err(MdError::ModelInitFailed(path.into())); }
        Ok(Self { model })
    }
    pub fn predict(&self, image: &Image) -> Result<Vec<ObbResult>, MdError> {
        let mut results = ffi::MDObbResults { data: ptr::null_mut(), size: 0 };
        let status = unsafe { ffi::md_obb_predict(&self.model, &image.raw, &mut results) };
        check_status(status)?;
        let out = if results.size > 0 && !results.data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(results.data, results.size as usize) };
            slice.iter().map(|r| ObbResult {
                xc: r.rotated_box.xc, yc: r.rotated_box.yc,
                width: r.rotated_box.width, height: r.rotated_box.height,
                angle: r.rotated_box.angle, label_id: r.label_id, score: r.score,
            }).collect()
        } else { Vec::new() };
        unsafe { ffi::md_free_obb_result(&mut results) };
        Ok(out)
    }
}
impl Drop for UltralyticsObb {
    fn drop(&mut self) { if !self.model.model_content.is_null() { unsafe { ffi::md_free_obb_model(&mut self.model) }; } }
}

/// 实例分割结果
#[derive(Debug, Clone)]
pub struct IsegResult {
    pub rect: Rect, pub label_id: i32, pub score: f32, pub mask_buffer: Vec<u8>,
    pub mask_shape: Vec<i32>,
}

/// Ultralytics 分割模型
#[derive(Debug)]
pub struct UltralyticsSeg {
    model: ffi::MDModel,
}
unsafe impl Send for UltralyticsSeg {}
unsafe impl Sync for UltralyticsSeg {}

impl UltralyticsSeg {
    pub fn new(path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::PathNotFound(path.into()))?;
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(), type_: ffi::MDModelType_Detection,
            format: ffi::MDModelFormat_ONNX, model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_instance_seg_model(&mut model, cpath.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() { return Err(MdError::ModelInitFailed(path.into())); }
        Ok(Self { model })
    }
    pub fn predict(&self, image: &Image) -> Result<Vec<IsegResult>, MdError> {
        let mut results = ffi::MDIsegResults { data: ptr::null_mut(), size: 0 };
        let status = unsafe { ffi::md_instance_seg_predict(&self.model, &image.raw, &mut results) };
        check_status(status)?;
        let out = if results.size > 0 && !results.data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(results.data, results.size as usize) };
            slice.iter().map(|r| {
                let buf = if !r.mask.buffer.is_null() {
                    unsafe { std::slice::from_raw_parts(r.mask.buffer as *const u8, r.mask.buffer_size as usize) }.to_vec()
                } else { Vec::new() };
                let shape = if !r.mask.shape.is_null() {
                    unsafe { std::slice::from_raw_parts(r.mask.shape, r.mask.num_dims as usize) }.to_vec()
                } else { Vec::new() };
                IsegResult {
                    rect: Rect { x: r.box_.x, y: r.box_.y, width: r.box_.width, height: r.box_.height },
                    label_id: r.label_id, score: r.score, mask_buffer: buf, mask_shape: shape,
                }
            }).collect()
        } else { Vec::new() };
        unsafe { ffi::md_free_instance_seg_result(&mut results) };
        Ok(out)
    }
}
impl Drop for UltralyticsSeg {
    fn drop(&mut self) { if !self.model.model_content.is_null() { unsafe { ffi::md_free_instance_seg_model(&mut self.model) }; } }
}

/// 姿态检测结果
#[derive(Debug, Clone)]
pub struct PoseResult {
    pub rect: Rect, pub score: f32, pub keypoints: Vec<Point3f>,
}

/// Ultralytics 姿态模型
#[derive(Debug)]
pub struct UltralyticsPose {
    model: ffi::MDModel,
}
unsafe impl Send for UltralyticsPose {}
unsafe impl Sync for UltralyticsPose {}

impl UltralyticsPose {
    pub fn new(path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::PathNotFound(path.into()))?;
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(), type_: ffi::MDModelType_Keypoint,
            format: ffi::MDModelFormat_ONNX, model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_keypoint_model(&mut model, cpath.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() { return Err(MdError::ModelInitFailed(path.into())); }
        Ok(Self { model })
    }
    pub fn predict(&self, image: &Image) -> Result<Vec<PoseResult>, MdError> {
        let mut results = ffi::MDKeyPointResults { data: ptr::null_mut(), size: 0 };
        let status = unsafe { ffi::md_keypoint_predict(&self.model, &image.raw, &mut results) };
        check_status(status)?;
        let out = if results.size > 0 && !results.data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(results.data, results.size as usize) };
            slice.iter().map(|r| {
                let kps = if !r.keypoints.is_null() {
                    let ks = unsafe { std::slice::from_raw_parts(r.keypoints, r.keypoints_size as usize) };
                    ks.iter().map(|kp| Point3f { x: kp.x, y: kp.y, z: kp.z }).collect()
                } else { Vec::new() };
                PoseResult {
                    rect: Rect { x: r.box_.x, y: r.box_.y, width: r.box_.width, height: r.box_.height },
                    score: r.score, keypoints: kps,
                }
            }).collect()
        } else { Vec::new() };
        unsafe { ffi::md_free_keypoint_result(&mut results) };
        Ok(out)
    }
}
impl Drop for UltralyticsPose {
    fn drop(&mut self) { if !self.model.model_content.is_null() { unsafe { ffi::md_free_keypoint_model(&mut self.model) }; } }
}
