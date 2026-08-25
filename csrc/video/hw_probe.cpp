#include "csrc/video/hw_probe.h"
#include <atomic>
#include <cstring>
#include <mutex>

// 通过 #ifdef 门控：ENABLE_FFMPEG / ENABLE_GSTREAMER 关闭时本翻译单元不引用对应原生头，
// 保证即使所在 .cpp 被统一 GLOB 编入库而对应后端未启用也能通过编译。
#ifdef ENABLE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
// 仅用 hwcontext.h 的 AV_HWDEVICE_TYPE_CUDA / av_hwdevice_ctx_create；
// 不包含 hwcontext_cuda.h（其依赖 CUDA 的 cuda.h，WITH_GPU=OFF 时缺 include 路径）。
#include <libavutil/hwcontext.h>
}
#endif
#ifdef ENABLE_GSTREAMER
#include <gst/gst.h>
#endif

namespace modeldeploy::video {

namespace {

#ifdef ENABLE_FFMPEG
// CUDA 设备上下文能否建立：avcodec 存在 ≠ 设备可用。建不出则剔除 cuvid/nvenc 项。
bool cuda_ctx_available() {
    AVBufferRef* ctx = nullptr;
    if (av_hwdevice_ctx_create(&ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) return false;
    if (ctx) av_buffer_unref(&ctx);
    return true;
}
#endif

#ifdef ENABLE_GSTREAMER
std::atomic<bool> g_gst_probe_ready{false};
std::mutex g_gst_probe_mtx;

// GStreamer 进程内一次性初始化（与 gst_decoder.cpp 的 md_gst_init_once 同款语义；
// gst_init_check 可重复调用，安全）。
bool gst_probe_init_once() {
    if (g_gst_probe_ready.load()) return true;
    std::lock_guard<std::mutex> lk(g_gst_probe_mtx);
    if (g_gst_probe_ready.load()) return true;
    GError* err = nullptr;
    if (gst_init_check(nullptr, nullptr, &err)) g_gst_probe_ready = true;
    if (err) g_error_free(err);
    return g_gst_probe_ready.load();
}
#endif

} // namespace

HwProbeResult probe_ffmpeg_hw() {
    HwProbeResult r;
#ifdef ENABLE_FFMPEG
    const bool cuda_ok = cuda_ctx_available();
    const char* decoders[] = {"h264_cuvid", "hevc_cuvid", "av1_cuvid",
                              "mjpeg_cuvid", "h264_vaapi", "hevc_vaapi"};
    for (const char* name : decoders) {
        if (!avcodec_find_decoder_by_name(name)) continue;
        if (std::strstr(name, "cuvid") && !cuda_ok) continue;  // CUDA 不可用则剔除
        r.decoders.emplace_back(name);
    }
    const char* encoders[] = {"h264_nvenc", "hevc_nvenc", "av1_nvenc",
                              "h264_vaapi", "hevc_vaapi", "h264_qsv", "hevc_qsv"};
    for (const char* name : encoders) {
        if (!avcodec_find_encoder_by_name(name)) continue;
        if (std::strstr(name, "nvenc") && !cuda_ok) continue;  // CUDA 不可用则剔除
        r.encoders.emplace_back(name);
    }
#endif
    return r;
}

HwProbeResult probe_gstreamer_hw() {
    HwProbeResult r;
#ifdef ENABLE_GSTREAMER
    if (!gst_probe_init_once()) return r;
    const char* names[] = {"nvh264enc", "nvh264dec", "nvv4l2decoder",
                           "vaapih264enc", "vaapih264dec"};
    for (const char* name : names) {
        GstElementFactory* f = gst_element_factory_find(name);
        if (!f) continue;
        if (std::strstr(name, "enc")) r.encoders.emplace_back(name);
        else r.decoders.emplace_back(name);  // dec / decoder
        gst_object_unref(f);
    }
#endif
    return r;
}

} // namespace modeldeploy::video
