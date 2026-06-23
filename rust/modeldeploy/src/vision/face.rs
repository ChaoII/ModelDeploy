use crate::error::{check_status, MdError};
use crate::ffi;
use crate::image::Image;
use crate::runtime::RuntimeOption;
use crate::types::{FaceDetection, Point3f, Rect};
use std::ffi::CString;
use std::ptr;

/// SCRFD 人脸检测模型封装
#[derive(Debug)]
pub struct Scrfd {
    model: ffi::MDModel,
}

unsafe impl Send for Scrfd {}
unsafe impl Sync for Scrfd {}

impl Scrfd {
    /// 加载人脸检测模型
    pub fn new(path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::PathNotFound(path.into()))?;
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(),
            type_: ffi::MDModelType_FACE,
            format: ffi::MDModelFormat_ONNX,
            model_content: ptr::null_mut(),
        };
        let status =
            unsafe { ffi::md_create_face_det_model(&mut model, cpath.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() {
            return Err(MdError::ModelInitFailed(path.into()));
        }
        Ok(Self { model })
    }

    /// 设置输入尺寸
    pub fn set_input_size(&mut self, width: i32, height: i32) -> Result<(), MdError> {
        let size = ffi::MDSize { width, height };
        let status = unsafe { ffi::md_set_face_det_input_size(&self.model, size) };
        check_status(status)
    }

    /// 推理
    pub fn predict(&self, image: &Image) -> Result<Vec<FaceDetection>, MdError> {
        let mut results = ffi::MDKeyPointResults {
            data: ptr::null_mut(),
            size: 0,
        };
        let status = unsafe { ffi::md_face_det_predict(&self.model, &image.raw, &mut results) };
        check_status(status)?;

        let faces = if results.size > 0 && !results.data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(results.data, results.size as usize) };
            slice
                .iter()
                .map(|r| {
                    let landmarks = if !r.keypoints.is_null() && r.keypoints_size > 0 {
                        let kp_slice = unsafe {
                            std::slice::from_raw_parts(r.keypoints, r.keypoints_size as usize)
                        };
                        kp_slice
                            .iter()
                            .map(|kp| Point3f {
                                x: kp.x,
                                y: kp.y,
                                z: kp.z,
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };

                    FaceDetection {
                        rect: Rect {
                            x: r.box_.x,
                            y: r.box_.y,
                            width: r.box_.width,
                            height: r.box_.height,
                        },
                        score: r.score,
                        landmarks,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        unsafe { ffi::md_free_face_det_result(&mut results) };
        Ok(faces)
    }

    /// 模型是否已初始化
    pub fn is_initialized(&self) -> bool {
        !self.model.model_content.is_null()
    }
}

impl Drop for Scrfd {
    fn drop(&mut self) {
        if !self.model.model_content.is_null() {
            unsafe { ffi::md_free_face_det_model(&mut self.model) };
        }
    }
}
