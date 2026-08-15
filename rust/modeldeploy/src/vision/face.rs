use crate::error::{check_status, MdError};
use crate::ffi;
use crate::image::Image;
use crate::runtime::RuntimeOption;
use crate::types::{FaceDetection, Point3f, Rect, InsightFaceResult};
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

impl Drop for Scrfd {
    fn drop(&mut self) {
        if !self.model.model_content.is_null() {
            unsafe { ffi::md_free_face_det_model(&mut self.model) };
        }
    }
}

/// insightface 综合人脸分析（det + 2D/3D landmark + recognition）
#[derive(Debug)]
pub struct InsightFaceAnalysis {
    model: ffi::MDModel,
}

unsafe impl Send for InsightFaceAnalysis {}
unsafe impl Sync for InsightFaceAnalysis {}

impl InsightFaceAnalysis {
    /// 加载 insightface 模型（det/rec/2d106/3d68 四模型路径）
    pub fn new(det_model: &str, rec_model: &str, lmk2d_model: &str, lmk3d_model: &str,
               option: &RuntimeOption) -> Result<Self, MdError> {
        let cd = CString::new(det_model).map_err(|_| MdError::PathNotFound(det_model.into()))?;
        let cr = CString::new(rec_model).map_err(|_| MdError::PathNotFound(rec_model.into()))?;
        let c2 = CString::new(lmk2d_model).map_err(|_| MdError::PathNotFound(lmk2d_model.into()))?;
        let c3 = CString::new(lmk3d_model).map_err(|_| MdError::PathNotFound(lmk3d_model.into()))?;
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(),
            type_: ffi::MDModelType_InsightFace,
            format: ffi::MDModelFormat_ONNX,
            model_content: ptr::null_mut(),
        };
        let status = unsafe {
            ffi::md_create_insightface_model(&mut model, cd.as_ptr(), cr.as_ptr(), c2.as_ptr(), c3.as_ptr(),
                                             &option.raw)
        };
        check_status(status)?;
        if model.model_content.is_null() {
            return Err(MdError::ModelInitFailed("insightface".into()));
        }
        Ok(Self { model })
    }

    /// 分析：检测 + 关键点 + 识别
    pub fn analyze(&self, image: &Image) -> Result<Vec<InsightFaceResult>, MdError> {
        let mut results = ffi::MDInsightFaceResults { data: ptr::null_mut(), size: 0 };
        let status = unsafe { ffi::md_insightface_analyze(&self.model, &image.raw, &mut results) };
        check_status(status)?;
        let out = if results.size > 0 && !results.data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(results.data, results.size as usize) };
            slice.iter().map(|r| InsightFaceResult {
                rect: Rect { x: r.box_.x, y: r.box_.y, width: r.box_.width, height: r.box_.height },
                score: r.score,
                kps: read_points(r.kps, r.kps_size),
                landmark_2d_106: read_points(r.landmark_2d_106, r.landmark_2d_106_size),
                landmark_3d_68: read_points(r.landmark_3d_68, r.landmark_3d_68_size),
                pose: r.pose,
                embedding: read_f32s(r.embedding, r.embedding_size),
            }).collect()
        } else {
            Vec::new()
        };
        unsafe { ffi::md_free_insightface_result(&mut results) };
        Ok(out)
    }

    /// 设置检测阈值
    pub fn set_det_thresh(&mut self, thresh: f32) {
        unsafe { ffi::md_insightface_set_det_thresh(&mut self.model, thresh) };
    }

    /// 是否初始化
    pub fn is_initialized(&self) -> bool {
        !self.model.model_content.is_null()
    }
}

impl Drop for InsightFaceAnalysis {
    fn drop(&mut self) {
        if !self.model.model_content.is_null() {
            unsafe { ffi::md_free_insightface_model(&mut self.model) };
        }
    }
}

fn read_points(p: *mut ffi::MDPoint3f, n: i32) -> Vec<Point3f> {
    if p.is_null() || n <= 0 { return Vec::new(); }
    let slice = unsafe { std::slice::from_raw_parts(p, n as usize) };
    slice.iter().map(|kp| Point3f { x: kp.x, y: kp.y, z: kp.z }).collect()
}

fn read_f32s(p: *mut f32, n: i32) -> Vec<f32> {
    if p.is_null() || n <= 0 { return Vec::new(); }
    unsafe { std::slice::from_raw_parts(p, n as usize) }.to_vec()
}
