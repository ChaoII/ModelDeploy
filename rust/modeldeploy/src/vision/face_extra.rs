use crate::error::{check_status, MdError};
use crate::ffi;
use crate::image::Image;
use crate::runtime::RuntimeOption;
use std::ffi::CString;
use std::ptr;

/// 人脸识别结果（embeddings）
#[derive(Debug, Clone)]
pub struct FaceRecResult {
    pub embedding: Vec<f32>,
}

/// 人脸识别模型
#[derive(Debug)]
pub struct FaceRec {
    model: ffi::MDModel,
}
unsafe impl Send for FaceRec {}
unsafe impl Sync for FaceRec {}

impl FaceRec {
    pub fn new(path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::PathNotFound(path.into()))?;
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(), type_: ffi::MDModelType_FACE,
            format: ffi::MDModelFormat_ONNX, model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_face_rec_model(&mut model, cpath.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() { return Err(MdError::ModelInitFailed(path.into())); }
        Ok(Self { model })
    }
    pub fn predict(&self, image: &Image) -> Result<FaceRecResult, MdError> {
        let mut result = ffi::MDFaceRecognizerResult { embedding: ptr::null_mut(), size: 0 };
        let status = unsafe { ffi::md_face_rec_predict(&self.model, &image.raw, &mut result) };
        check_status(status)?;
        let embed = if result.size > 0 && !result.embedding.is_null() {
            unsafe { std::slice::from_raw_parts(result.embedding, result.size as usize) }.to_vec()
        } else { Vec::new() };
        unsafe { ffi::md_free_face_rec_result(&mut result) };
        Ok(FaceRecResult { embedding: embed })
    }
}
impl Drop for FaceRec {
    fn drop(&mut self) { if !self.model.model_content.is_null() { unsafe { ffi::md_free_face_rec_model(&mut self.model) }; } }
}

/// 人脸年龄模型
#[derive(Debug)]
pub struct FaceAge {
    model: ffi::MDModel,
}
unsafe impl Send for FaceAge {}
unsafe impl Sync for FaceAge {}

impl FaceAge {
    pub fn new(path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::PathNotFound(path.into()))?;
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(), type_: ffi::MDModelType_FACE,
            format: ffi::MDModelFormat_ONNX, model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_face_age_model(&mut model, cpath.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() { return Err(MdError::ModelInitFailed(path.into())); }
        Ok(Self { model })
    }
    pub fn predict(&self, image: &Image) -> Result<i32, MdError> {
        let mut result: i32 = 0;
        let status = unsafe { ffi::md_face_age_predict(&self.model, &image.raw, &mut result) };
        check_status(status)?;
        Ok(result)
    }
}
impl Drop for FaceAge {
    fn drop(&mut self) { if !self.model.model_content.is_null() { unsafe { ffi::md_free_face_age_model(&mut self.model) }; } }
}

/// 人脸性别结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender { Female = 0, Male = 1 }

/// 人脸性别模型
#[derive(Debug)]
pub struct FaceGender {
    model: ffi::MDModel,
}
unsafe impl Send for FaceGender {}
unsafe impl Sync for FaceGender {}

impl FaceGender {
    pub fn new(path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::PathNotFound(path.into()))?;
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(), type_: ffi::MDModelType_FACE,
            format: ffi::MDModelFormat_ONNX, model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_face_gender_model(&mut model, cpath.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() { return Err(MdError::ModelInitFailed(path.into())); }
        Ok(Self { model })
    }
    pub fn predict(&self, image: &Image) -> Result<Gender, MdError> {
        let mut result = ffi::MDFaceGenderResult(0);
        let status = unsafe { ffi::md_face_gender_predict(&self.model, &image.raw, &mut result) };
        check_status(status)?;
        Ok(if result.0 == 1 { Gender::Male } else { Gender::Female })
    }
}
impl Drop for FaceGender {
    fn drop(&mut self) { if !self.model.model_content.is_null() { unsafe { ffi::md_free_face_gender_model(&mut self.model) }; } }
}

/// 人脸防伪结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntiSpoofResult { Real = 0, Fuzzy = 1, Spoof = 2 }

/// 人脸防伪 Pipeline（Det + 1st + 2nd）
#[derive(Debug)]
pub struct FaceAntiSpoofPipeline {
    model: ffi::MDModel,
}
unsafe impl Send for FaceAntiSpoofPipeline {}
unsafe impl Sync for FaceAntiSpoofPipeline {}

impl FaceAntiSpoofPipeline {
    pub fn new(det_path: &str, first_path: &str, second_path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cdet = CString::new(det_path).unwrap();
        let cfirst = CString::new(first_path).unwrap();
        let csecond = CString::new(second_path).unwrap();
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(), type_: ffi::MDModelType_PIPELINE,
            format: ffi::MDModelFormat_ONNX, model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_face_as_pipeline_model(&mut model, cdet.as_ptr(), cfirst.as_ptr(), csecond.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() { return Err(MdError::ModelInitFailed(det_path.into())); }
        Ok(Self { model })
    }
    pub fn predict(&self, image: &Image) -> Result<Vec<AntiSpoofResult>, MdError> {
        let mut results = ffi::MDFaceAsResults { data: ptr::null_mut(), size: 0 };
        let status = unsafe { ffi::md_face_as_pipeline_predict(&self.model, &image.raw, &mut results, 0.8, 0.3) };
        check_status(status)?;
        let out = if results.size > 0 && !results.data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(results.data, results.size as usize) };
            slice.iter().map(|r| {
                match r.0 { 0 => AntiSpoofResult::Real, 1 => AntiSpoofResult::Fuzzy, _ => AntiSpoofResult::Spoof }
            }).collect()
        } else { Vec::new() };
        unsafe { ffi::md_free_face_as_pipeline_result(&mut results) };
        Ok(out)
    }
}
impl Drop for FaceAntiSpoofPipeline {
    fn drop(&mut self) { if !self.model.model_content.is_null() { unsafe { ffi::md_free_face_as_pipeline_model(&mut self.model) }; } }
}

/// 人脸识别 Pipeline（Det + Rec）
#[derive(Debug)]
pub struct FaceRecPipeline {
    model: ffi::MDModel,
}
unsafe impl Send for FaceRecPipeline {}
unsafe impl Sync for FaceRecPipeline {}

impl FaceRecPipeline {
    pub fn new(det_path: &str, rec_path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cdet = CString::new(det_path).unwrap();
        let crec = CString::new(rec_path).unwrap();
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(), type_: ffi::MDModelType_PIPELINE,
            format: ffi::MDModelFormat_ONNX, model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_face_rec_pipeline_model(&mut model, cdet.as_ptr(), crec.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() { return Err(MdError::ModelInitFailed(det_path.into())); }
        Ok(Self { model })
    }
    pub fn predict(&self, image: &Image) -> Result<Vec<FaceRecResult>, MdError> {
        let mut results = ffi::MDFaceRecognizerResults { data: ptr::null_mut(), size: 0 };
        let status = unsafe { ffi::md_face_rec_pipeline_predict(&self.model, &image.raw, &mut results) };
        check_status(status)?;
        let out = if results.size > 0 && !results.data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(results.data, results.size as usize) };
            slice.iter().map(|r| {
                let embed = if r.size > 0 && !r.embedding.is_null() {
                    unsafe { std::slice::from_raw_parts(r.embedding, r.size as usize) }.to_vec()
                } else { Vec::new() };
                FaceRecResult { embedding: embed }
            }).collect()
        } else { Vec::new() };
        unsafe { ffi::md_free_face_rec_pipeline_result(&mut results) };
        Ok(out)
    }
}
impl Drop for FaceRecPipeline {
    fn drop(&mut self) { if !self.model.model_content.is_null() { unsafe { ffi::md_free_face_rec_pipeline_model(&mut self.model) }; } }
}
