use crate::ffi;
use std::ffi::CString;

/// RuntimeOption 构建器，用于配置推理后端和设备
#[derive(Debug)]
pub struct RuntimeOption {
    pub(crate) raw: ffi::MDRuntimeOption,
    // 持有 CString 生命周期，避免 raw 中的指针悬空
    trt_min_shape: Option<CString>,
    trt_opt_shape: Option<CString>,
    trt_max_shape: Option<CString>,
    trt_cache_path: Option<CString>,
    password: Option<CString>,
}

impl Default for RuntimeOption {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeOption {
    /// 创建默认 RuntimeOption
    pub fn new() -> Self {
        let raw = unsafe { ffi::md_create_default_runtime_option() };
        Self {
            raw,
            trt_min_shape: None,
            trt_opt_shape: None,
            trt_max_shape: None,
            trt_cache_path: None,
            password: None,
        }
    }

    /// 设置 GPU 设备
    pub fn gpu(mut self, device_id: i32) -> Self {
        self.raw.device_id = device_id;
        self.raw.device = ffi::MD_DEVICE_GPU;
        self
    }

    /// 设置 CPU
    pub fn cpu(mut self, threads: i32) -> Self {
        self.raw.device = ffi::MD_DEVICE_CPU;
        self.raw.cpu_thread_num = threads;
        self
    }

    /// 设置 CPU 线程数
    pub fn cpu_threads(mut self, n: i32) -> Self {
        self.raw.cpu_thread_num = n;
        self
    }

    /// 启用 ORT 后端
    pub fn ort_backend(mut self) -> Self {
        self.raw.backend = ffi::MD_BACKEND_ORT;
        self
    }

    /// 启用 MNN 后端
    pub fn mnn_backend(mut self) -> Self {
        self.raw.backend = ffi::MD_BACKEND_MNN;
        self
    }

    /// 启用 TRT 后端
    pub fn trt_backend(mut self) -> Self {
        self.raw.backend = ffi::MD_BACKEND_TRT;
        self
    }

    /// 启用 FP16
    pub fn fp16(mut self, enable: bool) -> Self {
        self.raw.enable_fp16 = if enable { 1 } else { 0 };
        self
    }

    /// 启用 ORT TensorRT EP
    pub fn enable_trt(mut self, enable: bool) -> Self {
        self.raw.enable_trt = if enable { 1 } else { 0 };
        self
    }

    /// 设置 TRT 最小 shape（格式 "images:1x3xHxW"）
    pub fn trt_min_shape(mut self, shape: &str) -> Self {
        self.trt_min_shape = Some(CString::new(shape).unwrap());
        if let Some(s) = &self.trt_min_shape {
            self.raw.trt_min_shape = s.as_ptr();
        }
        self
    }

    /// 设置 TRT 最优 shape
    pub fn trt_opt_shape(mut self, shape: &str) -> Self {
        self.trt_opt_shape = Some(CString::new(shape).unwrap());
        if let Some(s) = &self.trt_opt_shape {
            self.raw.trt_opt_shape = s.as_ptr();
        }
        self
    }

    /// 设置 TRT 最大 shape
    pub fn trt_max_shape(mut self, shape: &str) -> Self {
        self.trt_max_shape = Some(CString::new(shape).unwrap());
        if let Some(s) = &self.trt_max_shape {
            self.raw.trt_max_shape = s.as_ptr();
        }
        self
    }

    /// 设置 TRT engine 缓存路径
    pub fn trt_cache(mut self, path: &str) -> Self {
        self.trt_cache_path = Some(CString::new(path).unwrap());
        if let Some(s) = &self.trt_cache_path {
            self.raw.trt_engine_cache_path = s.as_ptr();
        }
        self
    }

    /// 设置模型密码（加密模型）
    pub fn password(mut self, pw: &str) -> Self {
        self.password = Some(CString::new(pw).unwrap());
        if let Some(s) = &self.password {
            self.raw.password = s.as_ptr();
        }
        self
    }

    /// 设置 ORT 日志级别（0=verbose, 4=fatal）
    pub fn ort_log_level(mut self, level: i32) -> Self {
        self.raw.ort_log_severity = level;
        self
    }
}

impl Drop for RuntimeOption {
    fn drop(&mut self) {
        // CString 字段自动释放
    }
}
