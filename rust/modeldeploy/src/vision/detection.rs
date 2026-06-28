use crate::error::{check_status, MdError};
use crate::ffi;
use crate::image::Image;
use crate::runtime::RuntimeOption;
use crate::types::{Detection, Rect};
use std::ffi::CString;
use std::ptr;

/// Ultralytics YOLO 检测模型封装
#[derive(Debug)]
pub struct UltralyticsDet {
    model: ffi::MDModel,
}

unsafe impl Send for UltralyticsDet {}
unsafe impl Sync for UltralyticsDet {}

impl UltralyticsDet {
    /// 加载检测模型
    pub fn new(path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::PathNotFound(path.into()))?;
        let mut model = ffi::MDModel {
            model_name: ptr::null_mut(),
            type_: ffi::MDModelType_Detection,
            format: ffi::MDModelFormat_ONNX,
            model_content: ptr::null_mut(),
        };
        let status =
            unsafe { ffi::md_create_detection_model(&mut model, cpath.as_ptr(), &option.raw) };
        check_status(status)?;
        if model.model_content.is_null() {
            return Err(MdError::ModelInitFailed(path.into()));
        }
        Ok(Self { model })
    }

    /// 设置输入尺寸
    pub fn set_input_size(&mut self, width: i32, height: i32) -> Result<(), MdError> {
        let size = ffi::MDSize { width, height };
        let status = unsafe { ffi::md_set_detection_input_size(&self.model, size) };
        check_status(status)
    }

    /// 推理
    pub fn predict(&self, image: &Image) -> Result<Vec<Detection>, MdError> {
        let mut results = ffi::MDDetectionResults {
            data: ptr::null_mut(),
            size: 0,
        };
        let status = unsafe { ffi::md_detection_predict(&self.model, &image.raw, &mut results) };
        check_status(status)?;

        let detections = if results.size > 0 && !results.data.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(results.data, results.size as usize) };
            slice
                .iter()
                .map(|r| Detection {
                    rect: Rect {
                        x: r.box_.x,
                        y: r.box_.y,
                        width: r.box_.width,
                        height: r.box_.height,
                    },
                    label_id: r.label_id,
                    score: r.score,
                    label_name: String::new(),
                })
                .collect()
        } else {
            Vec::new()
        };

        unsafe { ffi::md_free_detection_result(&mut results) };
        Ok(detections)
    }

    /// 推理并在图像上绘制结果，返回绘制后的图像
    pub fn predict_with_draw(
        &self,
        image: &Image,
        threshold: f64,
    ) -> Result<Image, MdError> {
        let mut results = ffi::MDDetectionResults {
            data: ptr::null_mut(),
            size: 0,
        };
        let status = unsafe { ffi::md_detection_predict(&self.model, &image.raw, &mut results) };
        check_status(status)?;

        let mut drawn = ffi::MDImage {
            width: 0,
            height: 0,
            channels: 0,
            data: ptr::null_mut(),
        };
        unsafe {
            ffi::md_draw_detection_result(
                &image.raw,
                &mut results,
                threshold,
                ptr::null(),
                &mut drawn,
            )
        };
        unsafe { ffi::md_free_detection_result(&mut results) };

        Ok(Image::from_raw(drawn))
    }

    /// 模型是否已初始化
    pub fn is_initialized(&self) -> bool {
        !self.model.model_content.is_null()
    }
}

impl Drop for UltralyticsDet {
    fn drop(&mut self) {
        if !self.model.model_content.is_null() {
            unsafe { ffi::md_free_detection_model(&mut self.model) };
        }
    }
}
