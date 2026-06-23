use crate::error::{check_status, MdError};
use crate::ffi;
use crate::image::Image;
use crate::runtime::RuntimeOption;
use std::ffi::CString;
use std::ptr;

/// OCR 检测结果
#[derive(Debug, Clone)]
pub struct OcrResult {
    pub text: String, pub score: f32, pub points: Vec<(i32, i32)>,
}

/// PaddleOCR 全模型（Det + Cls + Rec）
#[derive(Debug)]
pub struct PaddleOcr {
    model: ffi::MDModel,
}
unsafe impl Send for PaddleOcr {}
unsafe impl Sync for PaddleOcr {}

impl PaddleOcr {
    pub fn new(
        det_path: &str, cls_path: &str, rec_path: &str, dict_path: &str,
        option: &RuntimeOption,
    ) -> Result<Self, MdError> {
        let c_det = CString::new(det_path).unwrap();
        let c_cls = CString::new(cls_path).unwrap();
        let c_rec = CString::new(rec_path).unwrap();
        let c_dict = CString::new(dict_path).unwrap();
        let params = ffi::MDOCRModelParameters {
            det_model_file: c_det.as_ptr(), cls_model_file: c_cls.as_ptr(),
            rec_model_file: c_rec.as_ptr(), dict_path: c_dict.as_ptr(),
            max_side_len: 960, det_db_thresh: 0.3, det_db_box_thresh: 0.6,
            det_db_unclip_ratio: 1.5, det_db_score_mode: ptr::null(),
            use_dilation: 0, rec_batch_size: 6,
        };
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(), type_: ffi::MDModelType_OCR,
            format: ffi::MDModelFormat_ONNX, model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_ocr_model(&mut model, &params, &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() { return Err(MdError::ModelInitFailed(det_path.into())); }
        Ok(Self { model })
    }
    pub fn predict(&self, image: &Image) -> Result<Vec<OcrResult>, MdError> {
        let mut raw = image.raw.clone();
        let mut results = ffi::MDOCRResults { data: ptr::null_mut(), table_html: ptr::null_mut(), size: 0 };
        let status = unsafe { ffi::md_ocr_model_predict(&self.model, &mut raw, &mut results) };
        check_status(status)?;
        let out = if results.size > 0 && !results.data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(results.data, results.size as usize) };
            slice.iter().map(|r| {
                let text = if !r.text.is_null() {
                    unsafe { std::ffi::CStr::from_ptr(r.text).to_string_lossy().into_owned() }
                } else { String::new() };
                let pts = if !r.box_.data.is_null() && r.box_.size > 0 {
                    let pts_slice = unsafe { std::slice::from_raw_parts(r.box_.data, r.box_.size as usize) };
                    pts_slice.iter().map(|p| (p.x, p.y)).collect()
                } else { Vec::new() };
                OcrResult { text, score: r.score, points: pts }
            }).collect()
        } else { Vec::new() };
        unsafe { ffi::md_free_ocr_result(&mut results) };
        Ok(out)
    }
}
impl Drop for PaddleOcr {
    fn drop(&mut self) { if !self.model.model_content.is_null() { unsafe { ffi::md_free_ocr_model(&mut self.model) }; } }
}

/// OCR 识别（仅 Rec，不含 Det）
#[derive(Debug)]
pub struct OcrRecognition {
    model: ffi::MDModel,
}
unsafe impl Send for OcrRecognition {}
unsafe impl Sync for OcrRecognition {}

impl OcrRecognition {
    pub fn new(path: &str, dict_path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::PathNotFound(path.into()))?;
        let cdict = CString::new(dict_path).unwrap();
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(), type_: ffi::MDModelType_OCR,
            format: ffi::MDModelFormat_ONNX, model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_ocr_recognition_model(&mut model, cpath.as_ptr(), cdict.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() { return Err(MdError::ModelInitFailed(path.into())); }
        Ok(Self { model })
    }
    pub fn predict(&self, image: &Image) -> Result<OcrResult, MdError> {
        let mut result = ffi::MDOCRResult {
            box_: ffi::MDPolygon { data: ptr::null_mut(), size: 0 },
            text: ptr::null_mut(), score: 0.0,
            table_boxes: ffi::MDPolygon { data: ptr::null_mut(), size: 0 },
            table_structure: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_ocr_recognition_model_predict(&self.model, &image.raw, &mut result) };
        check_status(status)?;
        let text = if !result.text.is_null() {
            unsafe { std::ffi::CStr::from_ptr(result.text).to_string_lossy().into_owned() }
        } else { String::new() };
        let out = OcrResult { text, score: result.score, points: Vec::new() };
        unsafe { ffi::md_free_ocr_recognition_result(&mut result) };
        Ok(out)
    }
}
impl Drop for OcrRecognition {
    fn drop(&mut self) { if !self.model.model_content.is_null() { unsafe { ffi::md_free_ocr_recognition_model(&mut self.model) }; } }
}

/// LPR 车牌识别结果
#[derive(Debug, Clone)]
pub struct LprResult {
    pub rect: crate::types::Rect,
    pub plate_str: String, pub plate_color: String, pub score: f32,
    pub landmarks: Vec<(i32, i32)>,
}

/// LPR Pipeline（Det + Rec）
#[derive(Debug)]
pub struct LprPipeline {
    model: ffi::MDModel,
}
unsafe impl Send for LprPipeline {}
unsafe impl Sync for LprPipeline {}

impl LprPipeline {
    pub fn new(det_path: &str, rec_path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cdet = CString::new(det_path).unwrap();
        let crec = CString::new(rec_path).unwrap();
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(), type_: ffi::MDModelType_LPR,
            format: ffi::MDModelFormat_ONNX, model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_lpr_pipeline_model(&mut model, cdet.as_ptr(), crec.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() { return Err(MdError::ModelInitFailed(det_path.into())); }
        Ok(Self { model })
    }
    pub fn predict(&self, image: &Image) -> Result<Vec<LprResult>, MdError> {
        let mut results = ffi::MDLPRResults { data: ptr::null_mut(), size: 0 };
        let status = unsafe { ffi::md_lpr_pipeline_predict(&self.model, &image.raw, &mut results) };
        check_status(status)?;
        let out = if results.size > 0 && !results.data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(results.data, results.size as usize) };
            slice.iter().map(|r| {
                let plate = if !r.car_plate_str.is_null() { unsafe { std::ffi::CStr::from_ptr(r.car_plate_str).to_string_lossy().into_owned() } } else { String::new() };
                let color = if !r.car_plate_color.is_null() { unsafe { std::ffi::CStr::from_ptr(r.car_plate_color).to_string_lossy().into_owned() } } else { String::new() };
                let lm = if !r.landmarks.is_null() {
                    let ls = unsafe { std::slice::from_raw_parts(r.landmarks, r.landmarks_size as usize) };
                    ls.iter().map(|p| (p.x, p.y)).collect()
                } else { Vec::new() };
                LprResult {
                    rect: crate::types::Rect { x: r.box_.x, y: r.box_.y, width: r.box_.width, height: r.box_.height },
                    plate_str: plate, plate_color: color, score: r.score, landmarks: lm,
                }
            }).collect()
        } else { Vec::new() };
        unsafe { ffi::md_free_lpr_pipeline_result(&mut results) };
        Ok(out)
    }
}
impl Drop for LprPipeline {
    fn drop(&mut self) { if !self.model.model_content.is_null() { unsafe { ffi::md_free_lpr_pipeline_model(&mut self.model) }; } }
}

/// 行人属性结果
#[derive(Debug, Clone)]
pub struct AttributeResult {
    pub rect: crate::types::Rect, pub box_score: f32, pub attr_scores: Vec<f32>,
}

/// 行人属性模型
#[derive(Debug)]
pub struct PedestrianAttribute {
    model: ffi::MDModel,
}
unsafe impl Send for PedestrianAttribute {}
unsafe impl Sync for PedestrianAttribute {}

impl PedestrianAttribute {
    pub fn new(det_path: &str, cls_path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cdet = CString::new(det_path).unwrap();
        let ccls = CString::new(cls_path).unwrap();
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(), type_: ffi::MDModelType_PIPELINE,
            format: ffi::MDModelFormat_ONNX, model_content: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_create_attr_model(&mut model, cdet.as_ptr(), ccls.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() { return Err(MdError::ModelInitFailed(det_path.into())); }
        Ok(Self { model })
    }
    pub fn predict(&self, image: &Image) -> Result<Vec<AttributeResult>, MdError> {
        let mut results = ffi::MDAttributeResults { data: ptr::null_mut(), size: 0 };
        let status = unsafe { ffi::md_attr_model_predict(&self.model, &image.raw, &mut results) };
        check_status(status)?;
        let out = if results.size > 0 && !results.data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(results.data, results.size as usize) };
            slice.iter().map(|r| {
                let attrs = if !r.attr_scores.is_null() {
                    unsafe { std::slice::from_raw_parts(r.attr_scores, r.attr_scores_size as usize) }.to_vec()
                } else { Vec::new() };
                AttributeResult {
                    rect: crate::types::Rect { x: r.box_.x, y: r.box_.y, width: r.box_.width, height: r.box_.height },
                    box_score: r.box_score, attr_scores: attrs,
                }
            }).collect()
        } else { Vec::new() };
        unsafe { ffi::md_free_attr_result(&mut results) };
        Ok(out)
    }
}
impl Drop for PedestrianAttribute {
    fn drop(&mut self) { if !self.model.model_content.is_null() { unsafe { ffi::md_free_attr_model(&mut self.model) }; } }
}
