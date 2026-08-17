use crate::error::{check_status, MdError};
use crate::ffi;
use std::ffi::{CStr, CString};

/// 图像（对应 capi2 MDImageHandle，生命周期由本结构管理）
pub struct Image {
    pub(crate) handle: ffi::MDImageHandle,
    pub width: i32,
    pub height: i32,
}

impl Image {
    pub fn width(&self) -> i32 {
        self.width
    }

    pub fn height(&self) -> i32 {
        self.height
    }

    fn from_handle(handle: ffi::MDImageHandle) -> Result<Self, MdError> {
        if handle.is_null() {
            return Err(MdError::ImageDecode);
        }
        let mut w = 0;
        let mut h = 0;
        check_status(unsafe { ffi::md_image_size(handle, &mut w, &mut h) })?;
        Ok(Self {
            handle,
            width: w,
            height: h,
        })
    }

    /// 包装 predict_nv12 输出的绑定输入帧 ImageData（设备相关的 NV12 帧，库内不属主）。
    /// 生命周期归本结构管理（Drop 调 md_image_destroy，仅释放包装句柄，不碰输入缓冲）。
    pub fn from_device_frame(handle: ffi::MDImageHandle) -> Result<Self, MdError> {
        Self::from_handle(handle)
    }

    /// 取 NV12 帧的 Y/UV 平面指针 + 所在设备（仅对 NV12 帧有效；提供零拷贝外部访问）。
    /// 返回指针在图像句柄存活期间有效。
    pub fn plane_ptrs(&self) -> Result<(ffi::MDDevice, *const u8, *const u8), MdError> {
        let mut dev = ffi::MDDevice::CPU;
        let mut y: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut uv: *mut std::ffi::c_void = std::ptr::null_mut();
        check_status(unsafe { ffi::md_image_plane_ptrs(self.handle, &mut dev, &mut y, &mut uv) })?;
        Ok((dev, y as *const u8, uv as *const u8))
    }

    /// 从文件读取图像
    pub fn read(path: &str) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::InvalidArgument("path".into()))?;
        let mut handle = std::ptr::null_mut();
        check_status(unsafe { ffi::md_image_from_file(&mut handle, cpath.as_ptr()) })?;
        Self::from_handle(handle)
    }

    /// 从 BGR24 内存构造（引用，不拷贝）
    pub fn from_bgr24(data: &[u8], width: i32, height: i32) -> Result<Self, MdError> {
        let mut handle = std::ptr::null_mut();
        check_status(unsafe {
            ffi::md_image_from_bgr24(&mut handle, data.as_ptr() as *const _, width, height)
        })?;
        Self::from_handle(handle)
    }

    /// 从 RGB24 内存构造（拷贝并转换）
    pub fn from_rgb24(data: &[u8], width: i32, height: i32) -> Result<Self, MdError> {
        let mut handle = std::ptr::null_mut();
        check_status(unsafe {
            ffi::md_image_from_rgb24(&mut handle, data.as_ptr() as *const _, width, height)
        })?;
        Self::from_handle(handle)
    }

    /// 从 NV12 构造
    pub fn from_nv12(y: &[u8], uv: &[u8], width: i32, height: i32, step_y: i32, step_uv: i32) -> Result<Self, MdError> {
        let mut handle = std::ptr::null_mut();
        check_status(unsafe {
            ffi::md_image_from_nv12(
                &mut handle,
                y.as_ptr() as *const _,
                uv.as_ptr() as *const _,
                width,
                height,
                step_y,
                step_uv,
                ffi::MDDevice::CPU,
            )
        })?;
        Self::from_handle(handle)
    }

    /// 从编码数据解码
    pub fn from_encoded(bytes: &[u8]) -> Result<Self, MdError> {
        let mut handle = std::ptr::null_mut();
        check_status(unsafe {
            ffi::md_image_from_encoded(&mut handle, bytes.as_ptr() as *const _, bytes.len())
        })?;
        Self::from_handle(handle)
    }

    /// 深拷贝
    pub fn clone(&self) -> Result<Self, MdError> {
        let mut out = std::ptr::null_mut();
        check_status(unsafe { ffi::md_image_clone(self.handle, &mut out) })?;
        Self::from_handle(out)
    }

    /// 裁剪
    pub fn crop(&self, x: i32, y: i32, w: i32, h: i32) -> Result<Self, MdError> {
        let mut out = std::ptr::null_mut();
        check_status(unsafe { ffi::md_image_crop(self.handle, x, y, w, h, &mut out) })?;
        Self::from_handle(out)
    }

    /// 保存到文件
    pub fn save(&self, path: &str) -> Result<(), MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::InvalidArgument("path".into()))?;
        check_status(unsafe { ffi::md_image_save(self.handle, cpath.as_ptr()) })
    }

    /// 编码为字节
    pub fn encode(&self, ext: &str) -> Result<Vec<u8>, MdError> {
        let cext = CString::new(ext).map_err(|_| MdError::InvalidArgument("ext".into()))?;
        let mut buf: *const u8 = std::ptr::null();
        let mut n = 0usize;
        check_status(unsafe { ffi::md_image_encode(self.handle, cext.as_ptr(), &mut buf, &mut n) })?;
        if n == 0 || buf.is_null() {
            return Ok(Vec::new());
        }
        Ok(unsafe { std::slice::from_raw_parts(buf, n) }.to_vec())
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::md_image_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// C 字符串指针转 String
pub(crate) unsafe fn cstr_to_string(p: *const libc::c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}
