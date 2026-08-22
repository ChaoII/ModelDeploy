use crate::error::{check_status, MdError};
use crate::ffi::{self, MDBarcodeHandle};
use crate::image::{cstr_to_string, Image};
use crate::types::{BarcodeFormat, BarcodeResult};
use std::ptr;

/// 条码 / 二维码解码器（capi 单一分发，纯 CPU 无模型依赖）。
/// 无状态：detect 不推进内部状态，可并发/复用。
pub struct BarcodeDetector {
    handle: MDBarcodeHandle,
}

unsafe impl Send for BarcodeDetector {}
unsafe impl Sync for BarcodeDetector {}

impl BarcodeDetector {
    /// 创建条码解码器（formats 默认 FMT_ALL）。
    pub fn new() -> Result<Self, MdError> {
        let mut h = ptr::null_mut();
        check_status(unsafe { ffi::md_barcode_create(&mut h) })?;
        if h.is_null() {
            return Err(MdError::ModelInit("barcode".into()));
        }
        Ok(Self { handle: h })
    }

    /// 限定解码格式子集（FMT_* 位或）。
    pub fn set_formats(&self, formats: u32) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_barcode_set_formats(self.handle, formats) })
    }

    /// 检测并解码图片中的所有码。
    pub fn detect(&self, img: &Image) -> Result<Vec<BarcodeResult>, MdError> {
        // 阶段 1：容量查询（items==nullptr，不写入）
        let mut need: u32 = 0;
        check_status(unsafe {
            ffi::md_barcode_detect(self.handle, img.handle, ptr::null_mut(), &mut need)
        })?;

        // 阶段 2：分配 need 个项，正式解码写入
        let mut items = vec![ffi::MDBarcodeItem::default(); need as usize];
        let mut written = need;
        check_status(unsafe {
            ffi::md_barcode_detect(self.handle, img.handle, items.as_mut_ptr(), &mut written)
        })?;
        let written = (written as usize).min(items.len());

        let mut out = Vec::with_capacity(written);
        for it in items.iter().take(written) {
            let text = unsafe { cstr_to_string(it.text.as_ptr()) };
            let format = unsafe { cstr_to_string(it.format.as_ptr()) };
            out.push(BarcodeResult {
                text,
                format: BarcodeFormat::from_name(&format),
                quad: [
                    (it.quad[0], it.quad[1]),
                    (it.quad[2], it.quad[3]),
                    (it.quad[4], it.quad[5]),
                    (it.quad[6], it.quad[7]),
                ],
                score: it.score,
                is_qr: it.is_qr != 0,
            });
        }
        Ok(out)
    }
}

impl Drop for BarcodeDetector {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::md_barcode_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}
