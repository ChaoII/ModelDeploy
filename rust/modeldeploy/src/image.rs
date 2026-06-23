use crate::error::MdError;
use crate::ffi;
use std::ffi::CString;
use std::ptr;

/// 安全封装的图像对象，RAII 管理 MDImage 生命周期
#[derive(Debug)]
pub struct Image {
    pub(crate) raw: ffi::MDImage,
    owned: bool,
}

unsafe impl Send for Image {}
unsafe impl Sync for Image {}

impl Image {
    /// 从文件读取图像
    pub fn read(path: &str) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::ImageReadFailed(path.into()))?;
        let raw = unsafe { ffi::md_read_image(cpath.as_ptr()) };
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

    /// 从 NV12 数据创建图像
    pub fn from_nv12(data: &[u8], width: i32, height: i32) -> Result<Self, MdError> {
        let raw = unsafe { ffi::md_from_nv12_data_to_bgr24(data.as_ptr(), width, height) };
        if raw.data.is_null() || raw.width == 0 {
            return Err(MdError::ImageReadFailed("nv12_to_bgr24".into()));
        }
        Ok(Self { raw, owned: true })
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
        unsafe { ffi::md_save_image(&self.raw, cpath.as_ptr()) };
        Ok(())
    }

    /// 克隆图像（深拷贝）
    pub fn clone_image(&self) -> Result<Self, MdError> {
        let raw = unsafe { ffi::md_clone_image(&self.raw) };
        if raw.data.is_null() {
            return Err(MdError::OutOfMemory);
        }
        Ok(Self { raw, owned: true })
    }

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

pub mod nv12 {
    use crate::error::MdError;
    use crate::image::Image;

    pub fn to_bgr(
        y_plane: &[u8],
        uv_plane: &[u8],
        width: i32,
        height: i32,
    ) -> Result<Image, MdError> {
        // 合并 NV12 Y + UV 平面为连续 buffer
        let y_size = (width * height) as usize;
        let uv_size = y_size / 2;
        let mut nv12 = Vec::with_capacity(y_size + uv_size);
        nv12.extend_from_slice(y_plane);
        nv12.extend_from_slice(uv_plane);
        Image::from_nv12(&nv12, width, height)
    }
}
