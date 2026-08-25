use crate::error::{check_status, MdError};
use crate::ffi;
use std::ffi::CString;

/// 运行时选项（对应 capi MDOptionHandle，链式 setter）
pub struct RuntimeOption {
    pub(crate) handle: ffi::MDOptionHandle,
}

/// 把 Rust 字符串转成 CString（参数含内嵌 '\0' 视为不合法）
fn cstr(s: &str, what: &str) -> Result<CString, MdError> {
    CString::new(s).map_err(|_| MdError::InvalidArgument(format!("{what} 含内嵌 NUL")))
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
        let _ = unsafe { ffi::md_option_set_backend(self.handle, ffi::MDBackend::ORT) };
        self
    }

    pub fn use_mnn(&mut self) -> &mut Self {
        let _ = unsafe { ffi::md_option_set_backend(self.handle, ffi::MDBackend::MNN) };
        self
    }

    pub fn use_trt(&mut self) -> &mut Self {
        let _ = unsafe { ffi::md_option_set_backend(self.handle, ffi::MDBackend::TRT) };
        self
    }

    pub fn use_sophgo(&mut self) -> &mut Self {
        let _ = unsafe { ffi::md_option_set_backend(self.handle, ffi::MDBackend::SOPHGO) };
        self
    }

    pub fn set_device(&mut self, device: ffi::MDDevice, device_id: i32) -> Result<&mut Self, MdError> {
        check_status(unsafe { ffi::md_option_set_device(self.handle, device, device_id) })?;
        Ok(self)
    }

    pub fn set_backend(&mut self, backend: ffi::MDBackend) -> Result<&mut Self, MdError> {
        check_status(unsafe { ffi::md_option_set_backend(self.handle, backend) })?;
        Ok(self)
    }

    pub fn set_cpu_threads(&mut self, n: i32) -> Result<&mut Self, MdError> {
        check_status(unsafe { ffi::md_option_set_cpu_threads(self.handle, n) })?;
        Ok(self)
    }

    pub fn set_fp16(&mut self, enable: bool) -> Result<&mut Self, MdError> {
        check_status(unsafe { ffi::md_option_set_fp16(self.handle, enable as i32) })?;
        Ok(self)
    }

    pub fn set_external_stream(&mut self, stream: *mut std::ffi::c_void) -> Result<&mut Self, MdError> {
        check_status(unsafe { ffi::md_option_set_external_stream(self.handle, stream) })?;
        Ok(self)
    }

    pub fn set_password(&mut self, pwd: &str) -> Result<&mut Self, MdError> {
        let pwd = cstr(pwd, "密码")?;
        check_status(unsafe { ffi::md_option_set_password(self.handle, pwd.as_ptr()) })?;
        Ok(self)
    }

    pub fn set_model_path(&mut self, path: &str, pwd: &str) -> Result<&mut Self, MdError> {
        let path = cstr(path, "模型路径")?;
        let pwd = cstr(pwd, "密码")?;
        check_status(unsafe { ffi::md_option_set_model_path(self.handle, path.as_ptr(), pwd.as_ptr()) })?;
        Ok(self)
    }

    pub fn set_model_buffer(&mut self, data: &[u8], fmt: &str) -> Result<&mut Self, MdError> {
        let fmt = cstr(fmt, "模型格式")?;
        check_status(unsafe {
            ffi::md_option_set_model_buffer(self.handle, data.as_ptr(), data.len(), fmt.as_ptr())
        })?;
        Ok(self)
    }

    pub fn set_config(&mut self, ns: &str, key: &str, val: &str) -> Result<&mut Self, MdError> {
        let ns = cstr(ns, "命名空间")?;
        let key = cstr(key, "键")?;
        let val = cstr(val, "值")?;
        check_status(unsafe { ffi::md_option_set_config(self.handle, ns.as_ptr(), key.as_ptr(), val.as_ptr()) })?;
        Ok(self)
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
