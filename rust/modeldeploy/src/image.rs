use crate::error::{check_status, MdError};
use crate::ffi;
use std::ffi::CString;
use std::ptr;

/// 安全封装的图像对象，RAII 管理 MDImage 生命周期
#[derive(Debug)]
pub struct Image {
    pub(crate) raw: ffi::MDImage,
    // 标记是否为 CAPI 分配的内存（需要 free）
    owned: bool,
}

unsafe impl Send for Image {}
unsafe impl Sync for Image {}

impl Image {
    /// 从文件读取图像
    pub fn read(path: &str) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::ImageReadFailed(path.into()))?;
        let mut raw = ffi::MDImage {
            width: 0,
            height: 0,
            channels: 0,
            data: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_read_image(cpath.as_ptr(), &mut raw) };
        check_status(status)?;
        if raw.data.is_null() || raw.width == 0 || raw.height == 0 {
            return Err(MdError::ImageReadFailed(path.into()));
        }
        Ok(Self { raw, owned: true })
    }

    /// 从 BGR24 数据创建图像（不拷贝数据）
    pub fn from_bgr24(data: &[u8], width: i32, height: i32) -> Self {
        let raw = ffi::MDImage {
            width,
            height,
            channels: 3,
            data: data.as_ptr() as *mut u8,
        };
        Self { raw, owned: false }
    }

    /// 从 CAPI 分配的原始 MDImage 创建（标记为 owned）
    pub(crate) fn from_raw(raw: ffi::MDImage) -> Self {
        Self { raw, owned: true }
    }

    /// 图像宽度
    pub fn width(&self) -> i32 {
        self.raw.width
    }

    /// 图像高度
    pub fn height(&self) -> i32 {
        self.raw.height
    }

    /// 图像通道数
    pub fn channels(&self) -> i32 {
        self.raw.channels
    }

    /// 原始像素数据
    pub fn data(&self) -> &[u8] {
        if self.raw.data.is_null() {
            return &[];
        }
        let len = (self.raw.width * self.raw.height * self.raw.channels) as usize;
        unsafe { std::slice::from_raw_parts(self.raw.data, len) }
    }

    /// 保存图像到文件
    pub fn save(&self, path: &str) -> Result<(), MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::FileOpenFailed(path.into()))?;
        let status = unsafe { ffi::md_save_image(&self.raw, cpath.as_ptr()) };
        check_status(status)
    }

    /// 克隆图像（深拷贝）
    pub fn clone_image(&self) -> Result<Self, MdError> {
        let mut raw = ffi::MDImage {
            width: 0,
            height: 0,
            channels: 0,
            data: ptr::null_mut(),
        };
        let status = unsafe { ffi::md_clone_image(&self.raw, &mut raw) };
        check_status(status)?;
        Ok(Self { raw, owned: true })
    }

    /// 检查 CAPI 是否需要释放
    pub(crate) fn is_owned(&self) -> bool {
        self.owned
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        if self.owned && !self.raw.data.is_null() {
            unsafe { ffi::md_free_image(&mut self.raw) };
        }
    }
}

impl Clone for Image {
    fn clone(&self) -> Self {
        if self.owned {
            self.clone_image().unwrap()
        } else {
            Self {
                raw: self.raw.clone(),
                owned: false,
            }
        }
    }
}
