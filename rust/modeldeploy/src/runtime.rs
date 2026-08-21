use crate::error::MdError;
use crate::ffi;
use std::ffi::CString;

/// 运行时选项（对应 capi MDOptionHandle，链式 setter）
pub struct RuntimeOption {
    pub(crate) handle: ffi::MDOptionHandle,
}

impl RuntimeOption {
    /// 创建默认选项（CPU + ORT）
    pub fn new() -> Result<Self, MdError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { ffi::md_option_create(&mut handle) };
        crate::error::check_status(status)?;
        Ok(Self { handle })
    }

    pub fn use_ort(&mut self) -> &mut Self {
        unsafe { ffi::md_option_set_backend(self.handle, ffi::MDBackend::ORT) };
        self
    }

    pub fn use_mnn(&mut self) -> &mut Self {
        unsafe { ffi::md_option_set_backend(self.handle, ffi::MDBackend::MNN) };
        self
    }

    pub fn use_trt(&mut self) -> &mut Self {
        unsafe { ffi::md_option_set_backend(self.handle, ffi::MDBackend::TRT) };
        self
    }

    pub fn use_sophgo(&mut self) -> &mut Self {
        unsafe { ffi::md_option_set_backend(self.handle, ffi::MDBackend::SOPHGO) };
        self
    }

    pub fn set_device(&mut self, device: ffi::MDDevice) -> &mut Self {
        unsafe { ffi::md_option_set_device(self.handle, device) };
        self
    }

    pub fn set_cpu_threads(&mut self, n: i32) -> &mut Self {
        unsafe { ffi::md_option_set_cpu_threads(self.handle, n) };
        self
    }

    pub fn set_fp16(&mut self, enable: bool) -> &mut Self {
        unsafe { ffi::md_option_set_fp16(self.handle, enable as i32) };
        self
    }

    pub fn set_trt_engine_path(&mut self, path: &str) -> &mut Self {
        if let Ok(c) = CString::new(path) {
            unsafe { ffi::md_option_set_trt_engine_path(self.handle, c.as_ptr()) };
        }
        self
    }
}

impl Drop for RuntimeOption {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::md_option_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}
