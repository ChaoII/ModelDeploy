#include "stream_hub.hpp"
#include <iostream>
#include <cstring>

// ── SharedSource ─────────────────────────────────────────────

SharedSource::SharedSource(std::string url, DecoderConfig cfg)
    : url_(std::move(url)), cfg_(std::move(cfg)) {
}

SharedSource::~SharedSource() {
    if (decoder_) {
        decoder_->stop();
        decoder_.reset();
    }
    std::cout << "[SharedSource] Released: " << url_ << std::endl;
}

std::string SharedSource::init_error() const {
    return init_error_;
}

bool SharedSource::start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return true;

    decoder_ = std::make_unique<StreamDecoder>(cfg_);
    decoder_->set_callback([this](const DecodedFrame& f) -> bool {
        return this->on_decoded(f);
    });
    if (!decoder_->open(url_)) {
        init_error_ = "Decoder open failed: " + url_;
        std::cerr << "[SharedSource] " << init_error_ << std::endl;
        started_ = false;
        return false;
    }
    if (!decoder_->start()) {
        init_error_ = "Decoder start failed";
        std::cerr << "[SharedSource] " << init_error_ << std::endl;
        started_ = false;
        return false;
    }
    initialized_ = true;
    std::cout << "[SharedSource] Started: " << url_ << std::endl;
    return true;
}

uint64_t SharedSource::subscribe(FrameCallback cb) {
    uint64_t token = next_token_++;
    {
        std::lock_guard<std::mutex> lock(subs_mtx_);
        subscribers_[token] = std::move(cb);
    }
    // 第一个订阅者到达时才真正启动解码（避免文件源 EOF 抢跑）
    if (!started_.load()) {
        start();
    }
    return token;
}

void SharedSource::unsubscribe(uint64_t token) {
    std::lock_guard<std::mutex> lock(subs_mtx_);
    subscribers_.erase(token);
}

size_t SharedSource::subscriber_count() {
    std::lock_guard<std::mutex> lock(subs_mtx_);
    return subscribers_.size();
}

bool SharedSource::on_decoded(const DecodedFrame& f) {
    if (!started_.load()) return false;

    width_ = f.width;
    height_ = f.height;

    // 一次性把 NV12 紧凑拷贝（行连续）到 shared frame
    auto frame = std::make_shared<BroadcastFrame>();
    frame->width = f.width;
    frame->height = f.height;
    frame->pts = f.pts;

    size_t y_size = static_cast<size_t>(f.height) * f.width;
    size_t uv_size = y_size / 2;
    frame->nv12_data.resize(y_size + uv_size);
    uint8_t* dst = frame->nv12_data.data();
    for (int row = 0; row < f.height; ++row)
        std::memcpy(dst + row * f.width, f.y_plane + row * f.y_step, f.width);
    int uv_h = f.height / 2;
    for (int row = 0; row < uv_h; ++row)
        std::memcpy(dst + y_size + row * f.width, f.uv_plane + row * f.uv_step, f.width);

    // 拷贝订阅者列表（避免回调持锁）
    std::vector<FrameCallback> snap;
    {
        std::lock_guard<std::mutex> lock(subs_mtx_);
        snap.reserve(subscribers_.size());
        for (auto& [_, cb] : subscribers_) snap.push_back(cb);
    }

    // 派发：订阅者的回调必须是非阻塞的（只把帧 push 到自己的本地队列）
    for (auto& cb : snap) {
        try { cb(frame); } catch (...) {}
    }
    return true;
}

// ── StreamHub ────────────────────────────────────────────────

static bool is_network_stream(const std::string& url) {
    return url.find("rtsp://") == 0 || url.find("rtmp://") == 0 ||
           url.find("http://") == 0 || url.find("https://") == 0 ||
           url.find("udp://") == 0 || url.find("tcp://") == 0;
}

std::string StreamHub::make_key(const std::string& url, const DecoderConfig& cfg) {
    return url + "|" + cfg.rtsp_transport + "|" + cfg.hw_accel;
}

std::shared_ptr<SharedSource> StreamHub::acquire(const std::string& url, const DecoderConfig& cfg) {
    // 仅对网络流做共享；本地文件每路独立读取（避免 EOF 抢跑）
    if (!is_network_stream(url)) {
        auto src = std::make_shared<SharedSource>(url, cfg);
        std::cout << "[StreamHub] File source (independent): " << url << std::endl;
        return src;
    }

    std::string key = make_key(url, cfg);
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = sources_.find(key);
    if (it != sources_.end()) {
        if (auto sp = it->second.lock()) {
            std::cout << "[StreamHub] Reused source: " << key
                      << " (subscribers=" << sp->subscriber_count() << ")" << std::endl;
            return sp;
        }
        sources_.erase(it);
    }

    auto src = std::make_shared<SharedSource>(url, cfg);
    // 不立即 start：等第一个 subscribe 调用时才启动解码
    sources_[key] = src;
    std::cout << "[StreamHub] New source: " << key << std::endl;
    return src;
}
