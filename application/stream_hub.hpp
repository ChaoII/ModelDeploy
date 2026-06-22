#pragma once
#include <string>
#include <memory>
#include <unordered_map>
#include <map>
#include <mutex>
#include <atomic>
#include <functional>
#include <vector>

#include "config.hpp"
#include "stream_decoder.hpp"

/// 帧广播：解码线程产生 NV12 帧，分发给所有订阅者
struct BroadcastFrame {
    std::vector<uint8_t> nv12_data; // 紧凑 NV12 buffer（持有数据生命周期）
    int width = 0, height = 0;
    int64_t pts = 0;
};

/// 共享视频源：单一解码器 + 多订阅者
class SharedSource : public std::enable_shared_from_this<SharedSource> {
public:
    using FrameCallback = std::function<void(const std::shared_ptr<BroadcastFrame>&)>;

    SharedSource(std::string url, DecoderConfig cfg);
    ~SharedSource();

    /// 启动解码（首次订阅时调用；幂等）
    bool start();

    /// 注册订阅者，返回 token
    uint64_t subscribe(FrameCallback cb);

    /// 取消订阅
    void unsubscribe(uint64_t token);

    /// 当前订阅者数量
    size_t subscriber_count();

    /// 是否已成功打开解码器
    bool is_initialized() const { return initialized_.load(); }

    /// 初始化错误
    std::string init_error() const;

    /// 视频参数（订阅者需要知道宽高）
    int width() const { return width_; }
    int height() const { return height_; }

    /// 源流帧率
    int fps() const { return decoder_ ? decoder_->fps() : 25; }

private:
    bool on_decoded(const DecodedFrame& f);

    std::string url_;
    DecoderConfig cfg_;
    std::unique_ptr<StreamDecoder> decoder_;

    std::mutex subs_mtx_;
    std::unordered_map<uint64_t, FrameCallback> subscribers_;
    std::atomic<uint64_t> next_token_{1};

    std::atomic<bool> started_{false};
    std::atomic<bool> initialized_{false};
    std::string init_error_;
    int width_ = 0, height_ = 0;
};

/// 流共享中心：以 (url + decoder_cfg) 为 key 管理 SharedSource
/// 用 weak_ptr，最后一个订阅者退出时 SharedSource 自动析构
class StreamHub {
public:
    /// 获取或创建一个共享源
    std::shared_ptr<SharedSource> acquire(const std::string& url, const DecoderConfig& cfg);

private:
    static std::string make_key(const std::string& url, const DecoderConfig& cfg);

    std::map<std::string, std::weak_ptr<SharedSource>> sources_;
    std::mutex mtx_;
};
