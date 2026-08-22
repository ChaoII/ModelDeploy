use crate::error::{check_status, MdError};
use crate::ffi;
use crate::types::{ImageFormat, Plane};
use std::ffi::{CStr, CString};

/// 图像（对应 capi MDImageHandle，生命周期由本结构管理）
pub struct Image {
    pub(crate) handle: ffi::MDImageHandle,
    pub width: i32,
    pub height: i32,
    step_y: i32,
    step_uv: i32,
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
            step_y: 0,
            step_uv: 0,
        })
    }

    /// 包装 md_image_from_device_nv12 / md_model_predict 构造的绑定输入帧 ImageData（设备相关的 NV12 帧，库内不属主）。
    /// 生命周期归本结构管理（Drop 调 md_image_destroy，仅释放包装句柄，不碰输入缓冲）。
    #[allow(dead_code)]
    pub(crate) fn from_device_frame(handle: ffi::MDImageHandle) -> Result<Self, MdError> {
        Self::from_handle(handle)
    }

    /// 图像格式（经 md_image_info 查询）
    pub fn format(&self) -> ImageFormat {
        let mut t: i32 = 0;
        if unsafe {
            ffi::md_image_info(self.handle, &mut t, std::ptr::null_mut(), std::ptr::null_mut())
        } == ffi::MDStatus::OK
        {
            ImageFormat::from(t)
        } else {
            ImageFormat::Unknown
        }
    }

    /// 帧所在设备（经 md_image_info 查询）
    pub fn device(&self) -> ffi::MDDevice {
        let mut d = ffi::MDDevice::CPU;
        unsafe {
            ffi::md_image_info(self.handle, std::ptr::null_mut(), &mut d, std::ptr::null_mut());
        }
        d
    }

    /// 平面数量（经 md_image_info 查询）
    pub fn plane_count(&self) -> usize {
        let mut n: i32 = 0;
        if unsafe {
            ffi::md_image_info(self.handle, std::ptr::null_mut(), std::ptr::null_mut(), &mut n)
        } == ffi::MDStatus::OK
        {
            n.max(0) as usize
        } else {
            0
        }
    }

    /// 取第 i 个平面（仅 NV12 支持；越界或类型不符返回 Err）。
    /// 返回指针在图像句柄存活期间有效。
    pub fn plane(&self, i: usize) -> Result<Plane, MdError> {
        match self.format() {
            ImageFormat::NV12 => {}
            _ => return Err(MdError::UnsupportedType),
        }
        let mut dev = ffi::MDDevice::CPU;
        let mut y: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut uv: *mut std::ffi::c_void = std::ptr::null_mut();
        check_status(unsafe { ffi::md_image_plane_ptrs(self.handle, &mut dev, &mut y, &mut uv) })?;
        match i {
            0 => Ok(Plane {
                data: y as *const u8,
                step: if self.step_y > 0 { self.step_y } else { self.width },
            }),
            1 => Ok(Plane {
                data: uv as *const u8,
                step: if self.step_uv > 0 { self.step_uv } else { self.width },
            }),
            _ => Err(MdError::InvalidArgument("plane index".into())),
        }
    }

    /// 便捷方法：Y/UV 平面指针 + 所在设备（仅对 NV12/NV21 帧有效）。基于 plane()/device()。
    pub fn plane_ptrs(&self) -> Result<(ffi::MDDevice, *const u8, *const u8), MdError> {
        Ok((
            self.device(),
            self.plane(0)?.data,
            if self.plane_count() >= 2 {
                self.plane(1)?.data
            } else {
                std::ptr::null()
            },
        ))
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

    /// 从 NV12 构造（产真 NV12 两平面帧，库内拷入自有缓冲——安全，无需调用方保活）。
    /// 与 C# `FromNv12Data` 语义一致（图像类型为 NV12，走零拷贝 NV12 推理路径，不做 NV12→BGR 转换）。
    pub fn from_nv12(y: &[u8], uv: &[u8], width: i32, height: i32, step_y: i32, step_uv: i32) -> Result<Self, MdError> {
        let mut handle = std::ptr::null_mut();
        check_status(unsafe {
            ffi::md_image_from_nv12_owned(
                &mut handle,
                y.as_ptr() as *const _,
                uv.as_ptr() as *const _,
                width,
                height,
                step_y,
                step_uv,
            )
        })?;
        Self::from_handle(handle).map(|mut img| {
            img.step_y = step_y;
            img.step_uv = step_uv;
            img
        })
    }

    /// 从设备 NV12 两平面构造自描述 ImageData（零拷贝借用外部 y/uv 裸指针，库不拥有内存）。
    /// dev 指明帧所在设备（CPU/GPU/TPU）。返回的 Image 是统一 predict(ImageData) 单入口的输入，
    /// 也是可从平面指针访问的绑定输入帧。
    ///
    /// # Safety
    ///
    /// 调用方必须保证 `y`/`uv` 在返回的 `Image` 存活期内指向有效的内存（设备或主机内存取决于 `dev`），
    /// 且内容不被释放（库借用，不拥有）。
    pub unsafe fn from_device_nv12(
        y: *const u8,
        uv: *const u8,
        width: i32,
        height: i32,
        step_y: i32,
        step_uv: i32,
        dev: ffi::MDDevice,
    ) -> Result<Self, MdError> {
        let mut handle = std::ptr::null_mut();
        check_status(unsafe {
            ffi::md_image_from_device_nv12(
                &mut handle,
                y as *const _,
                uv as *const _,
                width,
                height,
                step_y,
                step_uv,
                dev,
            )
        })?;
        Self::from_handle(handle).map(|mut img| {
            img.step_y = step_y;
            img.step_uv = step_uv;
            img
        })
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
    #[allow(clippy::should_implement_trait)]
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
