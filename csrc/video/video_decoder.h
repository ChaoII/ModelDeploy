#pragma once

#include "csrc/video/video_frame.h"
#include "csrc/video/video_common.h"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/decode_pipeline.h"
#include <functional>
#include <memory>
#include <string>

namespace modeldeploy::video {

class DecoderBackend;

// SDK 视频解码门面（RAII 包装 DecodePipeline，后者包装 DecoderBackend）：
// 后端无关，接口不泄漏 FFmpeg/GStreamer 原生结构。提供同步 read_one_frame
// 与异步（set_callback + start）两条路径，二者共用缓冲池/背压/重连能力（异步）。
class MODELDEPLOY_CXX_EXPORT VideoDecoder {
public:
    // 按配置创建后端并返回门面；后端不可用时 set err 并返回 nullptr
    static std::shared_ptr<VideoDecoder> create(const VideoDecoderConfig& cfg,
                                                std::string* err = nullptr);
    ~VideoDecoder();

    bool open(const std::string& url, std::string* err = nullptr);
    // 抽下一帧到 *out（CPU NV12 + 毫秒时间戳）；失败/EOF 返回 false
    bool read_one_frame(VideoFrame* out, std::string* err = nullptr);

    using FrameCallback = std::function<void(VideoFrame&&)>;
    void set_callback(FrameCallback cb);

    bool start(std::string* err = nullptr);
    void stop();
    void set_device_only(bool v);

    State state() const;
    const VideoStats& stats() const;
    std::string last_error() const;
    int fps() const;
    int width() const;
    int height() const;
    void close();  // 幂等

    // 测试可观测：缓冲池命中（复用）次数；dropped/reconnect_count 经 stats() 读取
    uint64_t pool_hits() const;
    uint64_t pool_returns() const;

private:
    explicit VideoDecoder(std::shared_ptr<DecoderBackend> b, const VideoDecoderConfig& cfg);
    DecodePipeline pipeline_;
};

} // namespace modeldeploy::video
