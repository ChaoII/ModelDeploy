#pragma once
#include "csrc/video/video_codec_config.h"
#include "csrc/video/backend/encoder_backend.h"

// FFmpeg C 头仅在本后端内使用，绝不泄漏给门面/使用者侧。
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}
namespace modeldeploy::video {

// FFmpeg 软编后端（libx264）：CPU BGR → YUV420P 编码 → 封装输出。
// 本类经 factory 在 DLL 内以 make_shared 实例化，返回 shared_ptr<EncoderBackend> 给外部，
// 无需导出（与 EncoderBackend 接口一致，均不导出）。
class FfmpegEncoder : public EncoderBackend {
public:
    explicit FfmpegEncoder(const VideoEncoderConfig& cfg);
    ~FfmpegEncoder() override;

    bool runtime_available() const override;
    bool open(const std::string& url, int w, int h, int src_fps,
              const VideoEncoderConfig& c, std::string* err) override;
    bool encode(const modeldeploy::vision::ImageData& image, std::string* err) override;
    bool encode_from_gpu_nv12(const uint8_t* d_y, const uint8_t* d_uv, int w, int h,
                              std::string* err) override;
    bool encode_async(const modeldeploy::vision::ImageData& image) override;
    bool start_async(std::string* err) override;
    void stop_async() override;
    bool has_permanently_failed() const override;
    VideoStats& stats() override;
    std::string last_error() const override;
    void close() override;

private:
    bool init_encoder(int w, int h, int fps);
    bool open_output(const std::string& url);
    void cleanup();
    void set_err(std::string* err, const std::string& msg);

    VideoEncoderConfig cfg_;
    int w_ = 0, h_ = 0;
    int64_t pts_ = 0;
    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* enc_ = nullptr;
    AVStream* st_ = nullptr;
    SwsContext* sws_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* pkt_ = nullptr;
    bool header_ = false;
    bool opened_ = false;
    VideoStats stats_;
    std::string err_;
};

} // namespace modeldeploy::video
