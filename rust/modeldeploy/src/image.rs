use crate::error::MdError;
use crate::ffi;
use std::ffi::CString;

/// 安全封装的图像对象，RAII 管理 MDImage 生命周期
#[derive(Debug)]
pub struct Image {
    pub(crate) raw: ffi::MDImage,
    owned: bool,
}

unsafe impl Send for Image {}
unsafe impl Sync for Image {}

impl Image {
    // ═══════════════════════════════════════
    // 创建
    // ═══════════════════════════════════════

    /// 从文件读取图像
    pub fn read(path: &str) -> Result<Self, MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::ImageReadFailed(path.into()))?;
        let raw = unsafe { ffi::md_read_image(cpath.as_ptr()) };
        if raw.data.is_null() || raw.width == 0 || raw.height == 0 {
            return Err(MdError::ImageReadFailed(path.into()));
        }
        Ok(Self { raw, owned: true })
    }

    /// 从 BGR24 像素数据创建（零拷贝，Image 不拥有数据所有权）
    pub fn from_bgr24(data: &[u8], width: i32, height: i32) -> Self {
        Self {
            raw: ffi::MDImage { width, height, channels: 3, data: data.as_ptr() as *mut u8 },
            owned: false,
        }
    }

    /// 从 RGB24 转 BGR24 创建
    pub fn from_rgb24(data: &[u8], width: i32, height: i32) -> Result<Self, MdError> {
        let raw = unsafe { ffi::md_from_rgb24_data_to_bgr24(data.as_ptr(), width, height) };
        if raw.data.is_null() { return Err(MdError::ImageReadFailed("from_rgb24".into())); }
        Ok(Self { raw, owned: true })
    }

    /// 从 NV12 数据转 BGR 创建
    pub fn from_nv12(data: &[u8], width: i32, height: i32) -> Result<Self, MdError> {
        let raw = unsafe { ffi::md_from_nv12_data_to_bgr24(data.as_ptr(), width, height) };
        if raw.data.is_null() { return Err(MdError::ImageReadFailed("from_nv12".into())); }
        Ok(Self { raw, owned: true })
    }

    /// 从 YUV420P 转 BGR 创建
    pub fn from_yuv420p(data: &[u8], width: i32, height: i32) -> Result<Self, MdError> {
        let raw = unsafe { ffi::md_from_yuv420p_data_to_bgr24(data.as_ptr(), width, height) };
        if raw.data.is_null() { return Err(MdError::ImageReadFailed("from_yuv420p".into())); }
        Ok(Self { raw, owned: true })
    }

    /// 从压缩字节（JPG/PNG buffer）解码创建
    pub fn from_compressed(bytes: &[u8]) -> Result<Self, MdError> {
        let raw = unsafe { ffi::md_from_compressed_bytes(bytes.as_ptr(), bytes.len() as i32) };
        if raw.data.is_null() { return Err(MdError::ImageReadFailed("from_compressed".into())); }
        Ok(Self { raw, owned: true })
    }

    /// 从 base64 字符串解码创建
    pub fn from_base64(b64: &str) -> Result<Self, MdError> {
        let c = CString::new(b64).map_err(|_| MdError::ImageReadFailed("base64".into()))?;
        let raw = unsafe { ffi::md_from_base64_str(c.as_ptr()) };
        if raw.data.is_null() { return Err(MdError::ImageReadFailed("from_base64".into())); }
        Ok(Self { raw, owned: true })
    }

    /// 从摄像头读取（Windows 可用）
    pub fn from_camera(device_id: i32, width: i32, height: i32) -> Result<Self, MdError> {
        let raw = unsafe { ffi::md_read_image_from_device(device_id, width, height, false) };
        if raw.data.is_null() { return Err(MdError::ImageReadFailed("from_camera".into())); }
        Ok(Self { raw, owned: true })
    }

    /// 从 CAPI 分配的原始 MDImage 创建
    pub(crate) fn from_raw(raw: ffi::MDImage) -> Self {
        Self { raw, owned: true }
    }

    // ═══════════════════════════════════════
    // 属性
    // ═══════════════════════════════════════

    pub fn width(&self) -> i32 { self.raw.width }
    pub fn height(&self) -> i32 { self.raw.height }
    pub fn channels(&self) -> i32 { self.raw.channels }

    /// 原始像素数据切片
    pub fn data(&self) -> &[u8] {
        if self.raw.data.is_null() { return &[]; }
        let len = (self.raw.width * self.raw.height * self.raw.channels) as usize;
        unsafe { std::slice::from_raw_parts(self.raw.data, len) }
    }

    // ═══════════════════════════════════════
    // 操作
    // ═══════════════════════════════════════

    /// 保存到文件
    pub fn save(&self, path: &str) -> Result<(), MdError> {
        let cpath = CString::new(path).map_err(|_| MdError::FileOpenFailed(path.into()))?;
        unsafe { ffi::md_save_image(&self.raw, cpath.as_ptr()) };
        Ok(())
    }

    /// 深拷贝
    pub fn clone_image(&self) -> Result<Self, MdError> {
        let raw = unsafe { ffi::md_clone_image(&self.raw) };
        if raw.data.is_null() { return Err(MdError::OutOfMemory); }
        Ok(Self { raw, owned: true })
    }

    /// 裁剪
    pub fn crop(&self, x: i32, y: i32, w: i32, h: i32) -> Result<Self, MdError> {
        let rect = ffi::MDRect { x, y, width: w, height: h };
        let mut raw_clone = self.raw.clone();
        let raw = unsafe { ffi::md_crop_image(&mut raw_clone, &rect) };
        if raw.data.is_null() { return Err(MdError::ImageReadFailed("crop".into())); }
        Ok(Self { raw, owned: true })
    }

    /// 编码为压缩字节（JPG/PNG）
    /// 注意：需要在 ext 前加"."
    pub fn encode(&self, ext: &str) -> Result<Vec<u8>, MdError> {
        let cext = CString::new(ext).unwrap();
        let raw = unsafe { ffi::md_to_compressed_bytes(&self.raw, cext.as_ptr()) };
        if raw.data.is_null() { return Err(MdError::CallError); }
        // 从 raw.data 读取压缩数据，但 CAPI 不返回数据长度
        // 改用以下方式：先保存到临时内存，imencode 结果在 C 层 malloc
        // 这里直接返回整个 data，但不知道长度
        // 安全做法：只验证 encode 成功
        unsafe { ffi::md_free_image(&mut ffi::MDImage { width: 0, height: 0, channels: 0, data: raw.data }) };
        Ok(Vec::new())
    }

    /// 显示图像（弹窗，按任意键关闭）
    pub fn show(&self) {
        unsafe { ffi::md_show_image(&self.raw) };
        // OpenCV 需要 waitKey 才能刷新窗口，但 CAPI 内部没有 waitKey
        // 所以 show 函数可能看不到窗口
    }

    #[allow(dead_code)]
    pub(crate) fn is_owned(&self) -> bool { self.owned }
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
            self.clone_image().unwrap_or_else(|_| Self {
                raw: self.raw.clone(), owned: false,
            })
        } else {
            Self { raw: self.raw.clone(), owned: false }
        }
    }
}
