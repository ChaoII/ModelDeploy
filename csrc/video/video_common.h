#pragma once
#include <string>
#include <cstdint>

namespace modeldeploy::video {
// 解码/编码后端
enum class CodecBackend { FFmpeg, GStreamer };
inline std::string backend_to_string(CodecBackend b) {
    return b == CodecBackend::FFmpeg ? "ffmpeg" : "gstreamer";
}
// 硬件加速策略
enum class HwAccel { Auto, None, Cuda, Vaapi, Sophgo };
inline std::string hwaccel_to_string(HwAccel h) {
    switch (h) {
        case HwAccel::Auto: return "auto";
        case HwAccel::None: return "none";
        case HwAccel::Cuda: return "cuda";
        case HwAccel::Vaapi: return "vaapi";
        case HwAccel::Sophgo: return "sophgo";
    }
    return "none";
}
// 解码会话生命周期状态
enum class State { Idle, Opening, Running, Reconnecting, Eof, Error, Closed };
inline const char* state_to_string(State s) {
    switch (s) {
        case State::Idle: return "idle";
        case State::Opening: return "opening";
        case State::Running: return "running";
        case State::Reconnecting: return "reconnecting";
        case State::Eof: return "eof";
        case State::Error: return "error";
        case State::Closed: return "closed";
    }
    return "?";
}
// 错误码：永久性失败（PermanentFailure）与可重连失败区分
enum class ErrorCode {
    Ok,
    OpenFailed,
    ReadFailed,
    EncodeFailed,
    BackendUnavailable,
    NotInitialized,
    InvalidArgument,
    PermanentFailure
};
// 编解码统计
struct VideoStats {
    uint64_t frames_in = 0;
    uint64_t frames_out = 0;
    uint64_t dropped = 0;
    double avg_decode_ms = 0.0;
    double avg_encode_ms = 0.0;
    uint64_t reconnect_count = 0;
    uint64_t error_count = 0;
};
} // namespace modeldeploy::video
