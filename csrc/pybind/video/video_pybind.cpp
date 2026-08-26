//
// Python 侧视频编解码绑定：modeldeploy.video。
// 暴露与 C++ 门面对齐的全功能（与 C API / C# / Rust 等价）：
//   - 枚举：CodecBackend / HwAccel / Backpressure / State
//   - 配置：VideoDecoderConfig（全字段）、VideoEncoderConfig（全字段，含编码专用）
//   - 能力：VideoCapabilities + query_video_capabilities()
//   - 解码：VideoDecoder（同步 read_frame + 异步 set_callback/start + 状态/统计/缓冲池可观测）
//   - 编码：VideoEncoder（encode / encode_async / start_async / 状态/统计）
// ImageData 复用 vision 子模块已注册类型（modeldeploy.vision.ImageData）。
// 仅 BUILD_VIDEO && BUILD_VISION 时注册。
//

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <utility>

#if defined(BUILD_VIDEO) && defined(BUILD_VISION)
#include "csrc/video/video_decoder.h"
#include "csrc/video/video_encoder.h"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/factory.h"
#include "pybind/utils/utils.h"

namespace py = pybind11;

namespace modeldeploy::video {

namespace {

// 把 C++ VideoStats 拷成 Python 对象（stats() 返回内部引用，需拷贝）。
py::object make_stats_obj(pybind11::module&, const VideoStats& s) {
    // 直接构造绑定的 VideoStats 值（含 7 个只读字段）。
    VideoStats copy = s;
    return py::cast(copy);
}

} // namespace

void bind_video(pybind11::module& m) {
    // 后端 / 硬件加速 / 背压策略 / 会话状态枚举。
    py::enum_<CodecBackend>(m, "CodecBackend")
        .value("Auto", CodecBackend::Auto)
        .value("FFmpeg", CodecBackend::FFmpeg)
        .value("GStreamer", CodecBackend::GStreamer)
        .export_values();
    py::enum_<HwAccel>(m, "HwAccel")
        .value("Auto", HwAccel::Auto)
        .value("None", HwAccel::None)
        .value("Cuda", HwAccel::Cuda)
        .value("Vaapi", HwAccel::Vaapi)
        .value("Sophgo", HwAccel::Sophgo)
        .export_values();
    py::enum_<Backpressure>(m, "Backpressure")
        .value("Block", Backpressure::Block)
        .value("Drop", Backpressure::Drop)
        .value("OverwriteOldest", Backpressure::OverwriteOldest)
        .export_values();
    py::enum_<State>(m, "State")
        .value("Idle", State::Idle)
        .value("Opening", State::Opening)
        .value("Running", State::Running)
        .value("Reconnecting", State::Reconnecting)
        .value("Eof", State::Eof)
        .value("Error", State::Error)
        .value("Closed", State::Closed)
        .export_values();

    // 编解码统计（只读）。stats() 返回的是拷贝，安全。
    py::class_<VideoStats>(m, "VideoStats")
        .def(py::init<>())
        .def_readonly("frames_in", &VideoStats::frames_in)
        .def_readonly("frames_out", &VideoStats::frames_out)
        .def_readonly("dropped", &VideoStats::dropped)
        .def_readonly("avg_decode_ms", &VideoStats::avg_decode_ms)
        .def_readonly("avg_encode_ms", &VideoStats::avg_encode_ms)
        .def_readonly("reconnect_count", &VideoStats::reconnect_count)
        .def_readonly("error_count", &VideoStats::error_count)
        .def("__repr__", [](const VideoStats& s) {
            return "VideoStats(frames_in=" + std::to_string(s.frames_in) +
                   ", frames_out=" + std::to_string(s.frames_out) +
                   ", dropped=" + std::to_string(s.dropped) +
                   ", avg_decode_ms=" + std::to_string(s.avg_decode_ms) +
                   ", avg_encode_ms=" + std::to_string(s.avg_encode_ms) +
                   ")";
        });

    // 能力探测结果（只读）。
    py::class_<VideoCodecCapabilities>(m, "VideoCapabilities")
        .def(py::init<>())
        .def_readonly("ffmpeg_available", &VideoCodecCapabilities::ffmpeg_available)
        .def_readonly("gstreamer_available", &VideoCodecCapabilities::gstreamer_available)
        .def_readonly("hw_decoders", &VideoCodecCapabilities::hw_decoders)
        .def_readonly("hw_encoders", &VideoCodecCapabilities::hw_encoders)
        .def("__repr__", [](const VideoCodecCapabilities& c) {
            std::string d, e;
            for (const auto& s : c.hw_decoders) { if (!d.empty()) d += ", "; d += s; }
            for (const auto& s : c.hw_encoders) { if (!e.empty()) e += ", "; e += s; }
            return "VideoCapabilities(ffmpeg=" + std::string(c.ffmpeg_available ? "yes" : "no") +
                   ", gstreamer=" + std::string(c.gstreamer_available ? "yes" : "no") +
                   ", hw_decoders=[" + d + "], hw_encoders=[" + e + "])";
        });
    m.def("query_video_capabilities", &query_video_capabilities,
          "探测当前编译/运行环境可用的视频后端与硬件编解码能力");

    using modeldeploy::vision::ImageData;

    // 解码配置：全字段。
    py::class_<VideoDecoderConfig>(m, "VideoDecoderConfig")
        .def(py::init<>())
        .def_readwrite("backend", &VideoDecoderConfig::backend,
                       "解码后端：Auto / FFmpeg / GStreamer")
        .def_readwrite("hw_accel", &VideoDecoderConfig::hw_accel,
                       "硬件加速：Auto / None / Cuda / Vaapi / Sophgo")
        .def_readwrite("device_only", &VideoDecoderConfig::device_only,
                       "True 时 GPU 设备内存直通，不做主机往返")
        .def_readwrite("async_queue_size", &VideoDecoderConfig::async_queue_size,
                       "异步解码队列容量（有界）")
        .def_readwrite("backpressure", &VideoDecoderConfig::backpressure,
                       "队列满策略：Block / Drop / OverwriteOldest")
        .def_readwrite("pooling", &VideoDecoderConfig::pooling,
                       "帧缓冲池开关（True 复用容器，减少分配）")
        .def_readwrite("reconnect_delay_ms", &VideoDecoderConfig::reconnect_delay_ms,
                       "断流重连间隔（毫秒）")
        .def_readwrite("max_reconnects", &VideoDecoderConfig::max_reconnects,
                       "最大重连次数")
        .def_readwrite("timeout_us", &VideoDecoderConfig::timeout_us,
                       "网络/打开超时（微秒）")
        .def_readwrite("rtsp_transport", &VideoDecoderConfig::rtsp_transport,
                       "RTSP 传输协议：tcp / udp");

    // 编码配置：全字段（含编码专用）。
    py::class_<VideoEncoderConfig>(m, "VideoEncoderConfig")
        .def(py::init<>())
        // 继承自 VideoCodecConfig 的共用字段
        .def_readwrite("backend", &VideoEncoderConfig::backend,
                       "编码后端：Auto / FFmpeg / GStreamer")
        .def_readwrite("hw_accel", &VideoEncoderConfig::hw_accel,
                       "硬件加速：Auto / None / Cuda / Vaapi / Sophgo")
        .def_readwrite("device_only", &VideoEncoderConfig::device_only,
                       "True 时 GPU 设备内存直通")
        .def_readwrite("async_queue_size", &VideoEncoderConfig::async_queue_size,
                       "异步编码队列容量（有界）")
        .def_readwrite("backpressure", &VideoEncoderConfig::backpressure,
                       "队列满策略：Block / Drop / OverwriteOldest")
        .def_readwrite("pooling", &VideoEncoderConfig::pooling,
                       "帧缓冲池开关")
        .def_readwrite("reconnect_delay_ms", &VideoEncoderConfig::reconnect_delay_ms,
                       "断流重连间隔（毫秒）")
        .def_readwrite("max_reconnects", &VideoEncoderConfig::max_reconnects,
                       "最大重连次数")
        .def_readwrite("timeout_us", &VideoEncoderConfig::timeout_us,
                       "网络/打开超时（微秒）")
        .def_readwrite("rtsp_transport", &VideoEncoderConfig::rtsp_transport,
                       "RTSP 传输协议：tcp / udp")
        // 编码专用字段
        .def_readwrite("fps", &VideoEncoderConfig::fps, "输出帧率，0=自动")
        .def_readwrite("bitrate_kbps", &VideoEncoderConfig::bitrate_kbps, "编码码率（kbps）")
        .def_readwrite("gop", &VideoEncoderConfig::gop, "关键帧间隔")
        .def_readwrite("codec", &VideoEncoderConfig::codec,
                       "编码器：auto/libx264/x264enc/h264_nvenc/nvh264enc/vaapih264enc")
        .def_readwrite("preset", &VideoEncoderConfig::preset, "编码预设（如 ultrafast）")
        .def_readwrite("format", &VideoEncoderConfig::format,
                       "输出容器：auto/mp4/flv/rtmp/rtsp")
        .def_readwrite("max_b_frames", &VideoEncoderConfig::max_b_frames, "最大 B 帧数")
        .def_readwrite("low_latency", &VideoEncoderConfig::low_latency, "低延迟模式")
        .def_readwrite("gpu_direct_input", &VideoEncoderConfig::gpu_direct_input,
                       "True 时 GPU 显存 NV12 直编（需 hw_accel=Cuda 且 nvenc/nvh264enc）");

    // 解码器：同步 + 异步 + 状态/统计 + 缓冲池可观测。
    py::class_<VideoDecoder, std::shared_ptr<VideoDecoder>>(m, "VideoDecoder")
        .def(py::init([](const VideoDecoderConfig& cfg) {
            std::string err;
            auto dec = VideoDecoder::create(cfg, &err);
            if (!dec) {
                throw std::runtime_error("VideoDecoder::create failed: " + err);
            }
            return dec;
        }), pybind11::arg("cfg") = VideoDecoderConfig{})
        .def("open", [](VideoDecoder& d, const std::filesystem::path& url) {
            std::string err;
            bool ok = d.open(url.string(), &err);
            if (!ok && !err.empty()) {
                throw std::runtime_error("open failed: " + err);
            }
            return ok;
        }, pybind11::arg("url"))
        // 同步抽一帧，返回 (ok, ImageData, pts_ms)；EOF/失败时 ok=False。
        .def("read_frame", [](VideoDecoder& d) {
            VideoFrame vf;
            bool ok = d.read_one_frame(&vf);
            return std::make_tuple(ok, vf.image, vf.pts_ms);
        })
        // 异步：注册回调（后台线程投递，持 GIL 调用 py cb(image, pts_ms)）。
        .def("set_callback", [](VideoDecoder& d, py::function cb) {
            if (!cb) {
                throw std::runtime_error("set_callback: a callable is required");
            }
            // 捕获 py::function 副本：执行在解码线程，须经 GIL 进入 Python。
            d.set_callback([cb = std::move(cb)](VideoFrame&& vf) mutable {
                py::gil_scoped_acquire acquire;
                cb(vf.image, vf.pts_ms);
            });
        }, pybind11::arg("cb"))
        .def("start", [](VideoDecoder& d) {
            std::string err;
            bool ok = d.start(&err);
            if (!ok && !err.empty()) {
                throw std::runtime_error("start failed: " + err);
            }
            return ok;
        })
        .def("stop", &VideoDecoder::stop)
        .def("set_device_only", &VideoDecoder::set_device_only, pybind11::arg("enable"))
        .def("close", &VideoDecoder::close)
        .def_property_readonly("width", &VideoDecoder::width)
        .def_property_readonly("height", &VideoDecoder::height)
        .def_property_readonly("fps", &VideoDecoder::fps)
        .def_property_readonly("state", &VideoDecoder::state)
        .def("last_error", &VideoDecoder::last_error)
        .def("stats", [](VideoDecoder& d) { return VideoStats(d.stats()); })
        .def("pool_hits", &VideoDecoder::pool_hits)
        .def("pool_returns", &VideoDecoder::pool_returns);

    // 编码器：编码 + 异步 + 状态/统计。输入为 ImageData（复用 vision 类型）。
    py::class_<VideoEncoder, std::shared_ptr<VideoEncoder>>(m, "VideoEncoder")
        .def(py::init([](const VideoEncoderConfig& cfg) {
            std::string err;
            auto enc = VideoEncoder::create(cfg, &err);
            if (!enc) {
                throw std::runtime_error("VideoEncoder::create failed: " + err);
            }
            return enc;
        }), pybind11::arg("cfg") = VideoEncoderConfig{})
        .def("open", [](VideoEncoder& e, const std::filesystem::path& url,
                        int w, int height, int src_fps) {
            std::string err;
            bool ok = e.open(url.string(), w, height, src_fps, &err);
            if (!ok && !err.empty()) {
                throw std::runtime_error("open failed: " + err);
            }
            return ok;
        }, pybind11::arg("url"), pybind11::arg("width"), pybind11::arg("height"),
           pybind11::arg("src_fps"))
        // 编码一帧：image 为可视化 ImageData（CPU BGR 或设备 NV12）。
        .def("encode", [](VideoEncoder& e, const ImageData& image, uint64_t pts_ms) {
            VideoFrame vf;
            vf.image = image;
            vf.pts_ms = pts_ms;
            std::string err;
            bool ok = e.encode(vf, &err);
            if (!ok && !err.empty()) {
                throw std::runtime_error("encode failed: " + err);
            }
            return ok;
        }, pybind11::arg("image"), pybind11::arg("pts_ms") = 0)
        .def("encode_async", [](VideoEncoder& e, const ImageData& image) {
            return e.encode_async(image);
        }, pybind11::arg("image"))
        .def("start_async", [](VideoEncoder& e) {
            std::string err;
            bool ok = e.start_async(&err);
            if (!ok && !err.empty()) {
                throw std::runtime_error("start_async failed: " + err);
            }
            return ok;
        })
        .def("stop_async", &VideoEncoder::stop_async)
        .def("has_permanently_failed", &VideoEncoder::has_permanently_failed)
        .def("close", &VideoEncoder::close)
        .def_property_readonly("state", &VideoEncoder::state)
        .def("last_error", &VideoEncoder::last_error)
        .def("stats", [](VideoEncoder& e) { return VideoStats(e.stats()); });
}

} // namespace modeldeploy::video
#endif
