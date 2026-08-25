#include "csrc/video/decode_pipeline.h"
#include <chrono>
#include <utility>

namespace modeldeploy::video {

DecodePipeline::DecodePipeline(std::shared_ptr<DecoderBackend> backend,
                               const VideoDecoderConfig& cfg)
    : backend_(std::move(backend)), cfg_(cfg), pooling_(cfg.pooling) {}

DecodePipeline::~DecodePipeline() { close(); }

bool DecodePipeline::open(const std::string& url, std::string* err) {
    url_ = url;
    state_.store(static_cast<int>(State::Opening));
    bool ok = backend_ ? backend_->open(url, err) : false;
    state_.store(static_cast<int>(ok ? State::Running : State::Error));
    return ok;
}

bool DecodePipeline::read_one_frame(VideoFrame* out, std::string* err) {
    if (!backend_ || !out) return false;
    state_.store(static_cast<int>(State::Running));
    bool ok = backend_->read_one_frame(out, err);
    if (ok) stats_.frames_out++;
    return ok;
}

void DecodePipeline::set_callback(FrameCallback cb) {
    std::lock_guard<std::mutex> lk(qmtx_);
    callback_ = std::move(cb);
}

bool DecodePipeline::start(std::string* err) {
    std::lock_guard<std::mutex> lk(qmtx_);
    if (started_) return true;
    if (!backend_ || state_.load() != static_cast<int>(State::Running)) {
        if (err) *err = "not-opened";
        return false;
    }
    stop_requested_ = false;
    started_ = true;
    decode_thread_ = std::thread(&DecodePipeline::decode_loop, this);
    deliver_thread_ = std::thread(&DecodePipeline::deliver_loop, this);
    return true;
}

void DecodePipeline::stop() {
    {
        std::lock_guard<std::mutex> lk(qmtx_);
        if (!started_ && !stop_requested_) return;
        stop_requested_ = true;
    }
    qcond_not_empty_.notify_all();
    qcond_not_full_.notify_all();
    if (decode_thread_.joinable()) decode_thread_.join();
    if (deliver_thread_.joinable()) deliver_thread_.join();
    {
        std::lock_guard<std::mutex> lk(qmtx_);
        started_ = false;
        drain_queue_locked();
    }
}

void DecodePipeline::drain_queue_locked() {
    // 停产后清空残留帧（交付线程在停止时应已排空；此处持 qmtx_ 兜底回收，锁序 qmtx→pool 单向）
    while (!queue_.empty()) {
        auto f = std::move(queue_.front());
        queue_.pop_front();
        release_frame(std::move(f));
    }
}

void DecodePipeline::set_device_only(bool v) {
    if (backend_) backend_->set_device_only(v);
}

State DecodePipeline::state() const { return static_cast<State>(state_.load()); }

const VideoStats& DecodePipeline::stats() const { return stats_; }

std::string DecodePipeline::last_error() const { return err_; }

int DecodePipeline::fps() const { return backend_ ? backend_->fps() : 0; }

int DecodePipeline::width() const { return backend_ ? backend_->width() : 0; }

int DecodePipeline::height() const { return backend_ ? backend_->height() : 0; }

uint64_t DecodePipeline::pool_hits() const {
    std::lock_guard<std::mutex> lk(pool_mtx_);
    return pool_hits_;
}

uint64_t DecodePipeline::pool_returns() const {
    std::lock_guard<std::mutex> lk(pool_mtx_);
    return pool_returns_;
}

void DecodePipeline::close() {
    stop();
    if (backend_) backend_->close();
    state_.store(static_cast<int>(State::Closed));
}

std::shared_ptr<VideoFrame> DecodePipeline::acquire_frame() {
    if (!pooling_) return std::make_shared<VideoFrame>();
    std::lock_guard<std::mutex> lk(pool_mtx_);
    if (!pool_free_.empty()) {
        auto f = std::move(pool_free_.front());
        pool_free_.pop_front();
        pool_hits_++;
        return f;
    }
    return std::make_shared<VideoFrame>();
}

void DecodePipeline::release_frame(std::shared_ptr<VideoFrame> f) {
    if (!f) return;
    f->image = vision::ImageData{};
    f->pts_ms = 0;
    if (!pooling_) return;  // 直接释放（shared_ptr 析构回收）
    std::lock_guard<std::mutex> lk(pool_mtx_);
    pool_returns_++;
    pool_free_.push_back(std::move(f));
}

void DecodePipeline::decode_loop() {
    int reconnect_attempts = 0;
    while (!stop_requested_.load()) {
        std::string err;
        auto f = acquire_frame();
        auto t0 = std::chrono::steady_clock::now();
        bool ok = backend_->read_one_frame(f.get(), &err);
        auto dt = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
        if (!ok) {
            release_frame(std::move(f));
            if (stop_requested_.load()) break;
            // EOF：正常结束，不重连
            if (backend_->last_error() == "eof") {
                state_.store(static_cast<int>(State::Eof));
                break;
            }
            // 瞬态失败 → 重连状态机（非 EOF、未永久失败）
            if (cfg_.max_reconnects <= 0) {
                state_.store(static_cast<int>(State::Error));
                err_ = backend_->last_error();
                break;
            }
            state_.store(static_cast<int>(State::Reconnecting));
            std::this_thread::sleep_for(
                std::chrono::milliseconds(cfg_.reconnect_delay_ms));
            if (stop_requested_.load()) break;
            backend_->close();
            std::string oerr;
            if (!backend_->open(url_, &oerr)) {
                reconnect_attempts++;
                if (reconnect_attempts >= cfg_.max_reconnects) {
                    state_.store(static_cast<int>(State::Error));
                    err_ = "max-reconnects: " + backend_->last_error();
                    break;
                }
                continue;
            }
            stats_.reconnect_count++;   // 成功即收敛
            reconnect_attempts = 0;
            state_.store(static_cast<int>(State::Running));
            continue;
        }
        // 成功解码一帧
        double dt_avg = dt;
        uint64_t n = stats_.frames_in;
        stats_.avg_decode_ms = (stats_.avg_decode_ms * static_cast<double>(n) + dt_avg) /
                               static_cast<double>(n + 1);
        stats_.frames_in++;
        reconnect_attempts = 0;
        push_frame(std::move(f));
    }
    qcond_not_empty_.notify_all();
    qcond_not_full_.notify_all();
}

bool DecodePipeline::push_frame(std::shared_ptr<VideoFrame> f) {
    std::shared_ptr<VideoFrame> to_recycle;
    bool accepted = true;
    {
        std::unique_lock<std::mutex> lk(qmtx_);
        switch (cfg_.backpressure) {
            case Backpressure::Block: {
                qcond_not_full_.wait(lk, [&] {
                    return stop_requested_.load() ||
                           static_cast<int>(queue_.size()) < cfg_.async_queue_size;
                });
                if (stop_requested_.load()) {
                    to_recycle = std::move(f);
                    accepted = false;
                    break;
                }
                queue_.push_back(std::move(f));
                break;
            }
            case Backpressure::Drop:
                if (static_cast<int>(queue_.size()) >= cfg_.async_queue_size) {
                    stats_.dropped++;
                    to_recycle = std::move(f);
                    accepted = false;
                } else {
                    queue_.push_back(std::move(f));
                }
                break;
            case Backpressure::OverwriteOldest:
                if (static_cast<int>(queue_.size()) >= cfg_.async_queue_size) {
                    stats_.dropped++;
                    to_recycle = std::move(queue_.front());
                    queue_.pop_front();
                }
                queue_.push_back(std::move(f));
                break;
        }
        qcond_not_empty_.notify_one();
    }
    if (to_recycle) release_frame(std::move(to_recycle));  // 丢弃的帧容器同样回池
    return accepted;
}

void DecodePipeline::deliver_loop() {
    while (true) {
        std::shared_ptr<VideoFrame> f;
        {
            std::unique_lock<std::mutex> lk(qmtx_);
            qcond_not_empty_.wait(lk, [&] {
                return stop_requested_.load() || !queue_.empty();
            });
            if (stop_requested_.load() && queue_.empty()) break;
            f = std::move(queue_.front());
            queue_.pop_front();
            qcond_not_full_.notify_all();
        }
        if (callback_) callback_(std::move(*f));
        stats_.frames_out++;
        release_frame(std::move(f));
    }
    qcond_not_full_.notify_all();
}

} // namespace modeldeploy::video

