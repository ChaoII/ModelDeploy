//
// 视频编解码 P/Invoke（NativeMethods 的 partial 拆分）。
// 对应 capi：md_video_*（见 capi/md_capi.h）。SDK 需以 BUILD_VIDEO=ON 构建。
//

using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy
{
    internal static partial class NativeMethods
    {
        // 异步解码回调（对应 capi MDVideoFrameCb）。
        // frame 归接收方所有，回调内/回调后由接收方负责释放（md_image_destroy 由上层封装处理）。
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void MDVideoFrameCb(IntPtr frame, ulong ptsMs, IntPtr userdata);

        #region 配置

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_create(out IntPtr cfg);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_video_config_destroy(IntPtr cfg);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_backend(IntPtr cfg, MDCodecBackend b);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_hw_accel(IntPtr cfg, MDHwAccel h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_backpressure(IntPtr cfg, MDBackpressure bp);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_device_only(IntPtr cfg, int enable);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_async_queue_size(IntPtr cfg, int n);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_pooling(IntPtr cfg, int enable);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_reconnect_delay_ms(IntPtr cfg, int ms);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_max_reconnects(IntPtr cfg, int n);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_timeout_us(IntPtr cfg, int us);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_rtsp_transport(IntPtr cfg,
string t);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_fps(IntPtr cfg, int fps);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_bitrate_kbps(IntPtr cfg, int kbps);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_gop(IntPtr cfg, int gop);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_codec(IntPtr cfg,
string codec);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_preset(IntPtr cfg,
string preset);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_format(IntPtr cfg,
string fmt);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_max_b_frames(IntPtr cfg, int n);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_low_latency(IntPtr cfg, int enable);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_config_set_gpu_direct_input(IntPtr cfg, int enable);

        #endregion

        #region 能力探测

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_capabilities_create(out IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_video_capabilities_destroy(IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_capabilities_ffmpeg(IntPtr h, out int out_);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_capabilities_gstreamer(IntPtr h, out int out_);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_capabilities_hw_decoder_count(IntPtr h, out UIntPtr n);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_capabilities_hw_decoder(IntPtr h, UIntPtr i, out IntPtr name);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_capabilities_hw_encoder_count(IntPtr h, out UIntPtr n);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_capabilities_hw_encoder(IntPtr h, UIntPtr i, out IntPtr name);

        #endregion

        #region 解码器

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_decoder_create(IntPtr cfg, out IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_video_decoder_destroy(IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_decoder_open(IntPtr h,
string url);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_decoder_read_frame(IntPtr h, out IntPtr img, out ulong ptsMs);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_decoder_set_callback(IntPtr h, MDVideoFrameCb cb, IntPtr userdata);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_decoder_start(IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_video_decoder_stop(IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_decoder_set_device_only(IntPtr h, int enable);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_decoder_state(IntPtr h, out MDVideoState st);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_decoder_stats(IntPtr h, out MDVideoStats stats);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr md_video_decoder_last_error(IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_decoder_size(IntPtr h, out int w, out int height, out int fps);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_video_decoder_close(IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_decoder_pool_hits(IntPtr h, out ulong out_);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_decoder_pool_returns(IntPtr h, out ulong out_);

        #endregion

        #region 编码器

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_encoder_create(IntPtr cfg, out IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_video_encoder_destroy(IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_encoder_open(IntPtr h,
string url, int w, int height, int srcFps);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_encoder_encode(IntPtr h, IntPtr img, ulong ptsMs);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_encoder_encode_async(IntPtr h, IntPtr img);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_encoder_start_async(IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_video_encoder_stop_async(IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_encoder_has_permanently_failed(IntPtr h, out int out_);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_encoder_state(IntPtr h, out MDVideoState st);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_video_encoder_stats(IntPtr h, out MDVideoStats stats);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr md_video_encoder_last_error(IntPtr h);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_video_encoder_close(IntPtr h);

        #endregion
    }
}
