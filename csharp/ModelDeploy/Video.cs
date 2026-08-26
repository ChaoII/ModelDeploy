using System;
using ModelDeploy.types_internal_c;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy
{
    /// <summary>视频解码/编码统计（对应 C++ VideoStats）。</summary>
    public readonly struct VideoStats
    {
        public ulong FramesIn { get; }
        public ulong FramesOut { get; }
        public ulong Dropped { get; }
        public double AvgDecodeMs { get; }
        public double AvgEncodeMs { get; }
        public ulong ReconnectCount { get; }
        public ulong ErrorCount { get; }

        public VideoStats(MDVideoStats s)
        {
            FramesIn = s.frames_in;
            FramesOut = s.frames_out;
            Dropped = s.dropped;
            AvgDecodeMs = s.avg_decode_ms;
            AvgEncodeMs = s.avg_encode_ms;
            ReconnectCount = s.reconnect_count;
            ErrorCount = s.error_count;
        }
    }

    /// <summary>解码/编码通用配置（覆盖 capi 全部字段）。创建设置后传给 VideoDecoder/VideoEncoder。</summary>
    public sealed class VideoConfig : IDisposable
    {
        internal IntPtr Handle { get; private set; }
        private bool _disposed;

        public VideoConfig()
        {
            var status = md_video_config_create(out var h);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"VideoConfig create failed: {BaseModel.GetLastError()}");
            Handle = h;
        }

        public VideoCodecBackend Backend { set { Chk(md_video_config_set_backend(Handle, (MDCodecBackend)(int)value), "backend"); } }
        public VideoHwAccel HwAccel { set { Chk(md_video_config_set_hw_accel(Handle, (MDHwAccel)(int)value), "hw_accel"); } }
        public VideoBackpressure Backpressure { set { Chk(md_video_config_set_backpressure(Handle, (MDBackpressure)(int)value), "backpressure"); } }
        public bool DeviceOnly { set { Chk(md_video_config_set_device_only(Handle, value ? 1 : 0), "device_only"); } }
        public int AsyncQueueSize { set { Chk(md_video_config_set_async_queue_size(Handle, value), "async_queue_size"); } }
        public bool Pooling { set { Chk(md_video_config_set_pooling(Handle, value ? 1 : 0), "pooling"); } }
        public int ReconnectDelayMs { set { Chk(md_video_config_set_reconnect_delay_ms(Handle, value), "reconnect_delay_ms"); } }
        public int MaxReconnects { set { Chk(md_video_config_set_max_reconnects(Handle, value), "max_reconnects"); } }
        public int TimeoutUs { set { Chk(md_video_config_set_timeout_us(Handle, value), "timeout_us"); } }
        public string RtspTransport { set { Chk(md_video_config_set_rtsp_transport(Handle, value), "rtsp_transport"); } }

        // 编码专用
        public int Fps { set { Chk(md_video_config_set_fps(Handle, value), "fps"); } }
        public int BitrateKbps { set { Chk(md_video_config_set_bitrate_kbps(Handle, value), "bitrate_kbps"); } }
        public int Gop { set { Chk(md_video_config_set_gop(Handle, value), "gop"); } }
        public string Codec { set { Chk(md_video_config_set_codec(Handle, value), "codec"); } }
        public string Preset { set { Chk(md_video_config_set_preset(Handle, value), "preset"); } }
        public string Format { set { Chk(md_video_config_set_format(Handle, value), "format"); } }
        public int MaxBFrames { set { Chk(md_video_config_set_max_b_frames(Handle, value), "max_b_frames"); } }
        public bool LowLatency { set { Chk(md_video_config_set_low_latency(Handle, value ? 1 : 0), "low_latency"); } }
        public bool GpuDirectInput { set { Chk(md_video_config_set_gpu_direct_input(Handle, value ? 1 : 0), "gpu_direct_input"); } }

        private void Chk(MDStatus st, string what)
        {
            if (st != MDStatus.MD_OK)
                throw new InvalidOperationException($"VideoConfig[{what}] failed: {BaseModel.GetLastError()}");
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(VideoConfig));
        }

        private void Touch()
        {
            ThrowIfDisposed();
            if (Handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(VideoConfig));
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (Handle != IntPtr.Zero) { md_video_config_destroy(Handle); Handle = IntPtr.Zero; }
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~VideoConfig() => Dispose();
    }

    /// <summary>运行环境视频能力探测（对应 capi md_video_capabilities_*）。</summary>
    public sealed class VideoCapabilities : IDisposable
    {
        private IntPtr _h;
        private bool _disposed;

        public VideoCapabilities()
        {
            var status = md_video_capabilities_create(out var h);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"VideoCapabilities create failed: {BaseModel.GetLastError()}");
            _h = h;
        }

        public bool FfmpegAvailable { get { md_video_capabilities_ffmpeg(_h, out var v); return v != 0; } }
        public bool GstreamerAvailable { get { md_video_capabilities_gstreamer(_h, out var v); return v != 0; } }

        public string[] HwDecoders
        {
            get
            {
                md_video_capabilities_hw_decoder_count(_h, out var n);
                var arr = new string[(int)n];
                for (ulong i = 0; i < (ulong)n; i++)
                {
                    md_video_capabilities_hw_decoder(_h, (UIntPtr)i, out var p);
                    arr[i] = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(p) ?? "";
                }
                return arr;
            }
        }

        public string[] HwEncoders
        {
            get
            {
                md_video_capabilities_hw_encoder_count(_h, out var n);
                var arr = new string[(int)n];
                for (ulong i = 0; i < (ulong)n; i++)
                {
                    md_video_capabilities_hw_encoder(_h, (UIntPtr)i, out var p);
                    arr[i] = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(p) ?? "";
                }
                return arr;
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_h != IntPtr.Zero) { md_video_capabilities_destroy(_h); _h = IntPtr.Zero; }
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~VideoCapabilities() => Dispose();
    }

    /// <summary>视频解码器（对齐 C++ video::VideoDecoder，全功能）。</summary>
    public sealed class VideoDecoder : IDisposable
    {
        private IntPtr _h;
        private bool _disposed;
        private NativeMethods.MDVideoFrameCb _cbDelegate; // 保活：native 可能在回调线程调用

        public VideoDecoder(VideoConfig cfg = null)
        {
            IntPtr cfgH = IntPtr.Zero;
            VideoConfig ownedCfg = null;
            if (cfg == null)
            {
                ownedCfg = new VideoConfig();
                cfgH = ownedCfg.Handle;
            }
            else
            {
                cfgH = cfg.Handle;
            }
            try
            {
                var status = md_video_decoder_create(cfgH, out var h);
                if (status != MDStatus.MD_OK)
                    throw new InvalidOperationException($"VideoDecoder create failed: {BaseModel.GetLastError()}");
                _h = h;
            }
            finally
            {
                ownedCfg?.Dispose();
            }
        }

        public bool Open(string url)
        {
            var status = md_video_decoder_open(_h, url);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"VideoDecoder open failed: {BaseModel.GetLastError()}");
            return true;
        }

        /// <summary>同步取一帧：返回 (VisionImage, pts_ms)。EOF 抛 InvalidOperationException；图片用完必须 Dispose。</summary>
        public (VisionImage Image, long PtsMs) ReadFrame()
        {
            var status = md_video_decoder_read_frame(_h, out var img, out var pts);
            if (status == MDStatus.MD_ERR_VIDEO_DECODE)
                throw new InvalidOperationException($"VideoDecoder read_frame eof/error: {BaseModel.GetLastError()}");
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"VideoDecoder read_frame failed: {BaseModel.GetLastError()}");
            return (VisionImage.FromHandle(img), (long)pts);
        }

        /// <summary>异步：注册回调（跨线程投递；frame 用完须 Dispose），随后 Start。帧为自有句柄。</summary>
        public void SetCallback(Action<VisionImage, long> cb)
        {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            _cbDelegate = (frame, pts, _) =>
            {
                cb(VisionImage.FromHandle(frame), (long)pts);
            };
            var status = md_video_decoder_set_callback(_h, _cbDelegate, IntPtr.Zero);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"VideoDecoder set_callback failed: {BaseModel.GetLastError()}");
        }

        public void Start()
        {
            var status = md_video_decoder_start(_h);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"VideoDecoder start failed: {BaseModel.GetLastError()}");
        }

        public void Stop()
        {
            md_video_decoder_stop(_h);
        }

        public void SetDeviceOnly(bool enable)
        {
            Chk(md_video_decoder_set_device_only(_h, enable ? 1 : 0));
        }

        public VideoState State { get { md_video_decoder_state(_h, out var s); return (VideoState)s; } }
        public VideoStats Stats { get { md_video_decoder_stats(_h, out var s); return new VideoStats(s); } }
        public string LastError
        {
            get
            {
                var p = md_video_decoder_last_error(_h);
                return p == IntPtr.Zero ? "" : System.Runtime.InteropServices.Marshal.PtrToStringAnsi(p) ?? "";
            }
        }

        public int Width { get { md_video_decoder_size(_h, out int w, out _, out _); return w; } }
        public int Height { get { md_video_decoder_size(_h, out _, out int h, out _); return h; } }
        public int Fps { get { md_video_decoder_size(_h, out _, out _, out int f); return f; } }

        public ulong PoolHits { get { md_video_decoder_pool_hits(_h, out var v); return v; } }
        public ulong PoolReturns { get { md_video_decoder_pool_returns(_h, out var v); return v; } }

        public void Close()
        {
            ThrowIfDisposed();
            md_video_decoder_close(_h);
        }

        private void Chk(MDStatus st)
        {
            if (st != MDStatus.MD_OK)
                throw new InvalidOperationException($"VideoDecoder failed: {BaseModel.GetLastError()}");
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(VideoDecoder));
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_h != IntPtr.Zero) { md_video_decoder_close(_h); md_video_decoder_destroy(_h); _h = IntPtr.Zero; }
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~VideoDecoder() => Dispose();
    }

    /// <summary>视频编码器（对齐 C++ video::VideoEncoder，全功能）。</summary>
    public sealed class VideoEncoder : IDisposable
    {
        private IntPtr _h;
        private bool _disposed;

        public VideoEncoder(VideoConfig cfg = null)
        {
            IntPtr cfgH = IntPtr.Zero;
            VideoConfig ownedCfg = null;
            if (cfg == null)
            {
                ownedCfg = new VideoConfig();
                cfgH = ownedCfg.Handle;
            }
            else
            {
                cfgH = cfg.Handle;
            }
            try
            {
                var status = md_video_encoder_create(cfgH, out var h);
                if (status != MDStatus.MD_OK)
                    throw new InvalidOperationException($"VideoEncoder create failed: {BaseModel.GetLastError()}");
                _h = h;
            }
            finally
            {
                ownedCfg?.Dispose();
            }
        }

        public bool Open(string url, int w, int height, int srcFps)
        {
            var status = md_video_encoder_open(_h, url, w, height, srcFps);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"VideoEncoder open failed: {BaseModel.GetLastError()}");
            return true;
        }

        /// <summary>编码一帧（CPU BGR / 设备 NV12 + 可选 pts_ms）。img 的生命周期须覆盖本次调用。</summary>
        public bool Encode(VisionImage img, long ptsMs = 0)
        {
            var status = md_video_encoder_encode(_h, img.Handle, (ulong)ptsMs);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"VideoEncoder encode failed: {BaseModel.GetLastError()}");
            return true;
        }

        public bool EncodeAsync(VisionImage img)
        {
            var status = md_video_encoder_encode_async(_h, img.Handle);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"VideoEncoder encode_async failed: {BaseModel.GetLastError()}");
            return true;
        }

        public void StartAsync()
        {
            var status = md_video_encoder_start_async(_h);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"VideoEncoder start_async failed: {BaseModel.GetLastError()}");
        }

        public void StopAsync()
        {
            md_video_encoder_stop_async(_h);
        }

        public bool HasPermanentlyFailed { get { md_video_encoder_has_permanently_failed(_h, out var v); return v != 0; } }
        public VideoState State { get { md_video_encoder_state(_h, out var s); return (VideoState)s; } }
        public VideoStats Stats { get { md_video_encoder_stats(_h, out var s); return new VideoStats(s); } }
        public string LastError
        {
            get
            {
                var p = md_video_encoder_last_error(_h);
                return p == IntPtr.Zero ? "" : System.Runtime.InteropServices.Marshal.PtrToStringAnsi(p) ?? "";
            }
        }

        public void Close()
        {
            ThrowIfDisposed();
            md_video_encoder_close(_h);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(VideoEncoder));
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_h != IntPtr.Zero) { md_video_encoder_close(_h); md_video_encoder_destroy(_h); _h = IntPtr.Zero; }
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~VideoEncoder() => Dispose();
    }
}
