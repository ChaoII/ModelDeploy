//! 视频编解码（对�?capi md_video_*，全功能：解码同步/异步/设备直通 + 编码）。
//! 帧跨边界采用�?有权转移�?：read_frame / 异步回调返回的自有 Image 用后自动释放（Drop）。

use crate::error::{check_status, MdError};
use crate::ffi;
use crate::image::Image;
use std::ffi::CString;
use std::ptr;

/// 视频编解码后端（对应 capi MDCodecBackend）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecBackend {
    Auto = 0,
    FFmpeg = 1,
    GStreamer = 2,
}

/// 视频硬件加速（对应 capi MDHwAccel）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwAccel {
    Auto = 0,
    None = 1,
    Cuda = 2,
    Vaapi = 3,
    Sophgo = 4,
}

/// 异步队列满时背压策略（对应 capi MDBackpressure）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backpressure {
    Block = 0,
    Drop = 1,
    OverwriteOldest = 2,
}

/// 会话状态（对应 capi MDVideoState）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoState {
    Idle = 0,
    Opening = 1,
    Running = 2,
    Reconnecting = 3,
    Eof = 4,
    Error = 5,
    Closed = 6,
}

/// 编解码统计（对应 C++ VideoStats）。
#[derive(Debug, Clone, Copy, Default)]
pub struct VideoStats {
    pub frames_in: u64,
    pub frames_out: u64,
    pub dropped: u64,
    pub avg_decode_ms: f64,
    pub avg_encode_ms: f64,
    pub reconnect_count: u64,
    pub error_count: u64,
}

impl From<ffi::MDVideoStats> for VideoStats {
    fn from(s: ffi::MDVideoStats) -> Self {
        Self {
            frames_in: s.frames_in,
            frames_out: s.frames_out,
            dropped: s.dropped,
            avg_decode_ms: s.avg_decode_ms,
            avg_encode_ms: s.avg_encode_ms,
            reconnect_count: s.reconnect_count,
            error_count: s.error_count,
        }
    }
}

/// 统一帧（对应 C++ video::VideoFrame）。
pub struct VideoFrame {
    pub image: Image,
    pub pts_ms: u64,
}

/// 解码/编码配置（对齐 capi 全部字段）。构建后传给 VideoDecoder/VideoEncoder。
pub struct VideoConfig {
    handle: ffi::MDVideoConfigHandle,
}

impl VideoConfig {
    /// 新建默认配置（FFmpeg + Auto 硬件）。
    pub fn new() -> Result<Self, MdError> {
        let mut h = ptr::null_mut();
        check_status(unsafe { ffi::md_video_config_create(&mut h) })?;
        Ok(Self { handle: h })
    }

    fn chk(st: ffi::MDStatus) -> Result<(), MdError> {
        check_status(st)
    }

    pub fn backend(&mut self, b: CodecBackend) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_backend(self.handle, to_backend(b)) })?;
        Ok(self)
    }
    pub fn hw_accel(&mut self, h: HwAccel) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_hw_accel(self.handle, to_hw(h)) })?;
        Ok(self)
    }
    pub fn backpressure(&mut self, bp: Backpressure) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_backpressure(self.handle, to_bp(bp)) })?;
        Ok(self)
    }
    pub fn device_only(&mut self, v: bool) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_device_only(self.handle, v as i32) })?;
        Ok(self)
    }
    pub fn async_queue_size(&mut self, n: i32) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_async_queue_size(self.handle, n) })?;
        Ok(self)
    }
    pub fn pooling(&mut self, v: bool) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_pooling(self.handle, v as i32) })?;
        Ok(self)
    }
    pub fn reconnect_delay_ms(&mut self, ms: i32) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_reconnect_delay_ms(self.handle, ms) })?;
        Ok(self)
    }
    pub fn max_reconnects(&mut self, n: i32) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_max_reconnects(self.handle, n) })?;
        Ok(self)
    }
    pub fn timeout_us(&mut self, us: i32) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_timeout_us(self.handle, us) })?;
        Ok(self)
    }
    pub fn rtsp_transport(&mut self, t: &str) -> Result<&mut Self, MdError> {
        let c = CString::new(t).map_err(|_| MdError::InvalidArgument("rtsp_transport".into()))?;
        Self::chk(unsafe { ffi::md_video_config_set_rtsp_transport(self.handle, c.as_ptr()) })?;
        Ok(self)
    }
    // 编码专用
    pub fn fps(&mut self, v: i32) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_fps(self.handle, v) })?;
        Ok(self)
    }
    pub fn bitrate_kbps(&mut self, v: i32) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_bitrate_kbps(self.handle, v) })?;
        Ok(self)
    }
    pub fn gop(&mut self, v: i32) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_gop(self.handle, v) })?;
        Ok(self)
    }
    pub fn codec(&mut self, v: &str) -> Result<&mut Self, MdError> {
        let c = CString::new(v).map_err(|_| MdError::InvalidArgument("codec".into()))?;
        Self::chk(unsafe { ffi::md_video_config_set_codec(self.handle, c.as_ptr()) })?;
        Ok(self)
    }
    pub fn preset(&mut self, v: &str) -> Result<&mut Self, MdError> {
        let c = CString::new(v).map_err(|_| MdError::InvalidArgument("preset".into()))?;
        Self::chk(unsafe { ffi::md_video_config_set_preset(self.handle, c.as_ptr()) })?;
        Ok(self)
    }
    pub fn format(&mut self, v: &str) -> Result<&mut Self, MdError> {
        let c = CString::new(v).map_err(|_| MdError::InvalidArgument("format".into()))?;
        Self::chk(unsafe { ffi::md_video_config_set_format(self.handle, c.as_ptr()) })?;
        Ok(self)
    }
    pub fn max_b_frames(&mut self, v: i32) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_max_b_frames(self.handle, v) })?;
        Ok(self)
    }
    pub fn low_latency(&mut self, v: bool) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_low_latency(self.handle, v as i32) })?;
        Ok(self)
    }
    pub fn gpu_direct_input(&mut self, v: bool) -> Result<&mut Self, MdError> {
        Self::chk(unsafe { ffi::md_video_config_set_gpu_direct_input(self.handle, v as i32) })?;
        Ok(self)
    }
}

impl Drop for VideoConfig {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::md_video_config_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

/// 运行环境视频能力探测（对应 capi md_video_capabilities_*）。
#[derive(Default)]
pub struct VideoCapabilities {
    handle: ffi::MDVideoCapabilitiesHandle,
}

impl VideoCapabilities {
    /// 探测当前编译/运行环境能力（需 BUILD_VIDEO 构建）。
    pub fn probe() -> Result<Self, MdError> {
        let mut h = ptr::null_mut();
        check_status(unsafe { ffi::md_video_capabilities_create(&mut h) })?;
        Ok(Self { handle: h })
    }

    pub fn ffmpeg_available(&self) -> Result<bool, MdError> {
        let mut v: i32 = 0;
        check_status(unsafe { ffi::md_video_capabilities_ffmpeg(self.handle, &mut v) })?;
        Ok(v != 0)
    }
    pub fn gstreamer_available(&self) -> Result<bool, MdError> {
        let mut v: i32 = 0;
        check_status(unsafe { ffi::md_video_capabilities_gstreamer(self.handle, &mut v) })?;
        Ok(v != 0)
    }
    pub fn hw_decoders(&self) -> Result<Vec<String>, MdError> {
        let mut n: usize = 0;
        check_status(unsafe { ffi::md_video_capabilities_hw_decoder_count(self.handle, &mut n) })?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut p = ptr::null();
            check_status(unsafe { ffi::md_video_capabilities_hw_decoder(self.handle, i, &mut p) })?;
            out.push(unsafe { crate::image::cstr_to_string(p) });
        }
        Ok(out)
    }
    pub fn hw_encoders(&self) -> Result<Vec<String>, MdError> {
        let mut n: usize = 0;
        check_status(unsafe { ffi::md_video_capabilities_hw_encoder_count(self.handle, &mut n) })?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut p = ptr::null();
            check_status(unsafe { ffi::md_video_capabilities_hw_encoder(self.handle, i, &mut p) })?;
            out.push(unsafe { crate::image::cstr_to_string(p) });
        }
        Ok(out)
    }
}

impl Drop for VideoCapabilities {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::md_video_capabilities_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

/// 视频解码器（对齐 C++ video::VideoDecoder，全功能）。
pub struct VideoDecoder {
    handle: ffi::MDVideoDecoderHandle,
    callback: Option<Box<dyn FnMut(VideoFrame)>>,
}

/// 异步解码回调转�?fn（跨线程投递；frame 归接收方所有，Drop 自动释放）。
unsafe extern "C" fn video_cb_trampoline(
    frame: ffi::MDImageHandle,
    pts_ms: u64,
    userdata: *mut std::ffi::c_void,
) {
    let cb = unsafe { &mut *(userdata as *mut Box<dyn FnMut(VideoFrame)>) };
    let image = match crate::image::Image::from_device_frame(frame) {
        Ok(img) => img,
        Err(_) => {
            unsafe { ffi::md_image_destroy(frame) };
            return;
        }
    };
    cb(VideoFrame { image, pts_ms });
}

impl VideoDecoder {
    /// 按配置创建解码器（cfg 可省略，用默认）。
    pub fn create(cfg: Option<&VideoConfig>) -> Result<Self, MdError> {
        let cfg_h = cfg.map(|c| c.handle).unwrap_or(ptr::null_mut());
        let mut h = ptr::null_mut();
        check_status(unsafe { ffi::md_video_decoder_create(cfg_h, &mut h) })?;
        if h.is_null() {
            return Err(MdError::UnsupportedBackend);
        }
        Ok(Self { handle: h, callback: None })
    }

    /// 打开视频/流（本地文件或 RTSP/RTMP）。
    pub fn open(&self, url: &str) -> Result<(), MdError> {
        let c = CString::new(url).map_err(|_| MdError::InvalidArgument("url".into()))?;
        check_status(unsafe { ffi::md_video_decoder_open(self.handle, c.as_ptr()) })
    }

    /// 同步取一帧：返回 (自有 Image, 时间戳毫秒)。EOF 返回 Err(MdError::VideoDecode(...))。
    pub fn read_frame(&self) -> Result<VideoFrame, MdError> {
        let mut img = ptr::null_mut();
        let mut pts: u64 = 0;
        check_status(unsafe { ffi::md_video_decoder_read_frame(self.handle, &mut img, &mut pts) })?;
        Ok(VideoFrame {
            image: crate::image::Image::from_device_frame(img)?,
            pts_ms: pts,
        })
    }

    /// 异步：注册回调（跨线程投递，帧为自有 Image，用后自动释放），随后 start()。
    pub fn set_callback<F>(&mut self, cb: F) -> Result<(), MdError>
    where
        F: FnMut(VideoFrame) + Send + 'static,
    {
        self.callback = Some(Box::new(cb));
        let box_ptr = self.callback.as_mut().unwrap() as *mut Box<dyn FnMut(VideoFrame)>
            as *mut std::ffi::c_void;
        check_status(unsafe {
            ffi::md_video_decoder_set_callback(self.handle, video_cb_trampoline, box_ptr)
        })
    }

    pub fn start(&self) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_video_decoder_start(self.handle) })
    }

    pub fn stop(&self) {
        unsafe { ffi::md_video_decoder_stop(self.handle) }
    }

    pub fn set_device_only(&self, v: bool) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_video_decoder_set_device_only(self.handle, v as i32) })
    }

    pub fn state(&self) -> Result<VideoState, MdError> {
        let mut s = ffi::MDVideoState::Idle;
        check_status(unsafe { ffi::md_video_decoder_state(self.handle, &mut s) })?;
        Ok(from_state(s))
    }

    pub fn stats(&self) -> Result<VideoStats, MdError> {
        let mut s = ffi::MDVideoStats::default();
        check_status(unsafe { ffi::md_video_decoder_stats(self.handle, &mut s) })?;
        Ok(s.into())
    }

    pub fn last_error(&self) -> String {
        let p = unsafe { ffi::md_video_decoder_last_error(self.handle) };
        if p.is_null() { String::new() } else { unsafe { crate::image::cstr_to_string(p) } }
    }

    pub fn size(&self) -> Result<(i32, i32, i32), MdError> {
        let (mut w, mut h, mut f): (i32, i32, i32) = (0, 0, 0);
        check_status(unsafe { ffi::md_video_decoder_size(self.handle, &mut w, &mut h, &mut f) })?;
        Ok((w, h, f))
    }

    pub fn pool_hits(&self) -> Result<u64, MdError> {
        let mut v: u64 = 0;
        check_status(unsafe { ffi::md_video_decoder_pool_hits(self.handle, &mut v) })?;
        Ok(v)
    }
    pub fn pool_returns(&self) -> Result<u64, MdError> {
        let mut v: u64 = 0;
        check_status(unsafe { ffi::md_video_decoder_pool_returns(self.handle, &mut v) })?;
        Ok(v)
    }

    pub fn close(&self) {
        unsafe { ffi::md_video_decoder_close(self.handle) }
    }
}

impl Drop for VideoDecoder {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::md_video_decoder_close(self.handle) };
            unsafe { ffi::md_video_decoder_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

/// 视频编码器（对齐 C++ video::VideoEncoder，全功能）。
pub struct VideoEncoder {
    handle: ffi::MDVideoEncoderHandle,
}

impl VideoEncoder {
    pub fn create(cfg: Option<&VideoConfig>) -> Result<Self, MdError> {
        let cfg_h = cfg.map(|c| c.handle).unwrap_or(ptr::null_mut());
        let mut h = ptr::null_mut();
        check_status(unsafe { ffi::md_video_encoder_create(cfg_h, &mut h) })?;
        if h.is_null() {
            return Err(MdError::UnsupportedBackend);
        }
        Ok(Self { handle: h })
    }

    pub fn open(&self, url: &str, w: i32, height: i32, src_fps: i32) -> Result<(), MdError> {
        let c = CString::new(url).map_err(|_| MdError::InvalidArgument("url".into()))?;
        check_status(unsafe { ffi::md_video_encoder_open(self.handle, c.as_ptr(), w, height, src_fps) })
    }

    /// 编码一帧（img 生命周期须覆盖本次调用；GPU 输入按 cfg 直通直编）。
    pub fn encode(&self, img: &Image, pts_ms: u64) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_video_encoder_encode(self.handle, img.handle, pts_ms) })
    }

    pub fn encode_async(&self, img: &Image) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_video_encoder_encode_async(self.handle, img.handle) })
    }

    pub fn start_async(&self) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_video_encoder_start_async(self.handle) })
    }

    pub fn stop_async(&self) {
        unsafe { ffi::md_video_encoder_stop_async(self.handle) }
    }

    pub fn has_permanently_failed(&self) -> Result<bool, MdError> {
        let mut v: i32 = 0;
        check_status(unsafe { ffi::md_video_encoder_has_permanently_failed(self.handle, &mut v) })?;
        Ok(v != 0)
    }

    pub fn state(&self) -> Result<VideoState, MdError> {
        let mut s = ffi::MDVideoState::Idle;
        check_status(unsafe { ffi::md_video_encoder_state(self.handle, &mut s) })?;
        Ok(from_state(s))
    }

    pub fn stats(&self) -> Result<VideoStats, MdError> {
        let mut s = ffi::MDVideoStats::default();
        check_status(unsafe { ffi::md_video_encoder_stats(self.handle, &mut s) })?;
        Ok(s.into())
    }

    pub fn last_error(&self) -> String {
        let p = unsafe { ffi::md_video_encoder_last_error(self.handle) };
        if p.is_null() { String::new() } else { unsafe { crate::image::cstr_to_string(p) } }
    }

    pub fn close(&self) {
        unsafe { ffi::md_video_encoder_close(self.handle) }
    }
}

impl Drop for VideoEncoder {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::md_video_encoder_close(self.handle) };
            unsafe { ffi::md_video_encoder_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

fn to_backend(b: CodecBackend) -> ffi::MDCodecBackend {
    match b {
        CodecBackend::Auto => ffi::MDCodecBackend::Auto,
        CodecBackend::FFmpeg => ffi::MDCodecBackend::FFmpeg,
        CodecBackend::GStreamer => ffi::MDCodecBackend::GStreamer,
    }
}
fn to_hw(h: HwAccel) -> ffi::MDHwAccel {
    match h {
        HwAccel::Auto => ffi::MDHwAccel::Auto,
        HwAccel::None => ffi::MDHwAccel::None,
        HwAccel::Cuda => ffi::MDHwAccel::Cuda,
        HwAccel::Vaapi => ffi::MDHwAccel::Vaapi,
        HwAccel::Sophgo => ffi::MDHwAccel::Sophgo,
    }
}
fn to_bp(bp: Backpressure) -> ffi::MDBackpressure {
    match bp {
        Backpressure::Block => ffi::MDBackpressure::Block,
        Backpressure::Drop => ffi::MDBackpressure::Drop,
        Backpressure::OverwriteOldest => ffi::MDBackpressure::OverwriteOldest,
    }
}
fn from_state(s: ffi::MDVideoState) -> VideoState {
    match s {
        ffi::MDVideoState::Idle => VideoState::Idle,
        ffi::MDVideoState::Opening => VideoState::Opening,
        ffi::MDVideoState::Running => VideoState::Running,
        ffi::MDVideoState::Reconnecting => VideoState::Reconnecting,
        ffi::MDVideoState::Eof => VideoState::Eof,
        ffi::MDVideoState::Error => VideoState::Error,
        ffi::MDVideoState::Closed => VideoState::Closed,
    }
}
