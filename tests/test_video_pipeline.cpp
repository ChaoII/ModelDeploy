#include "catch2/catch_test_macros.hpp"
#include "csrc/video/decode_pipeline.h"
#include "csrc/video/video_decoder.h"
#include "csrc/video/backend/decoder_backend.h"
#include <atomic>
#include <chrono>
#include <fstream>
#include <memory>
#include <thread>

using namespace modeldeploy::video;

namespace {

// 可注入故障的桩后端：不连网，驱动缓冲池/背压/重连状态机的确定性单测。
class FakeDecoderBackend : public DecoderBackend {
public:
    int frames_to_produce = 300;
    // 恰好在第 n 次 read 时返回一次"非 eof"瞬态失败（之后恢复产帧，驱动重连）
    int transient_read_at = -1;
    // 重连期间：第几次 open 失败后才成功（初始 open 恒成功）
    int reconnect_open_failures = 0;

    bool runtime_available() const override { return true; }
    bool open(const std::string&, std::string* err) override {
        opens_++;
        bool is_reconnect = opens_ > 1;
        if (is_reconnect && (opens_ - 1) <= reconnect_open_failures) {
            if (err) *err = "transient-open-fail";
            return false;
        }
        opened_ = true;
        return true;
    }
    bool read_one_frame(VideoFrame* out, std::string* err) override {
        if (!opened_) {
            if (err) *err = "not-opened";
            return false;
        }
        reads_++;
        if (reads_ == transient_read_at) {
            last_err_ = "transient-read-fail";
            if (err) *err = last_err_;
            return false;
        }
        if (produced_ >= frames_to_produce) {
            last_err_ = "eof";
            if (err) *err = last_err_;
            return false;
        }
        out->pts_ms = static_cast<uint64_t>(produced_);
        out->image = modeldeploy::vision::ImageData{};
        produced_++;
        return true;
    }
    void set_callback(FrameCallback) override {}
    bool start(std::string*) override { return true; }
    void stop() override {}
    void set_device_only(bool) override {}
    int fps() const override { return 25; }
    int width() const override { return 320; }
    int height() const override { return 240; }
    VideoStats& stats() override { return stats_; }
    std::string last_error() const override { return last_err_; }
    void close() override { opened_ = false; }

    int produced_count() const { return produced_.load(); }
    int open_count() const { return opens_.load(); }
    int read_count() const { return reads_.load(); }

private:
    std::atomic<int> produced_{0};
    std::atomic<int> opens_{0};
    std::atomic<int> reads_{0};
    bool opened_ = false;
    VideoStats stats_;
    std::string last_err_;
};

template <class Pred>
bool wait_until(Pred p, int timeout_ms = 8000) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (p()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return p();
}

}  // namespace

// === 缓冲池 ===
TEST_CASE("pipeline 帧缓冲池：容器复用（池命中>0）", "[video][pipeline][pool]") {
    auto fake = std::make_shared<FakeDecoderBackend>();
    fake->frames_to_produce = 300;
    VideoDecoderConfig cfg;
    cfg.pooling = true;
    cfg.backpressure = Backpressure::Block;
    cfg.async_queue_size = 64;
    DecodePipeline pipe(fake, cfg);
    std::atomic<int> delivered{0};
    pipe.set_callback([&](VideoFrame&&) { delivered++; });
    std::string err;
    REQUIRE(pipe.open("fake://src", &err));
    REQUIRE(pipe.start(&err));
    REQUIRE(wait_until([&] { return delivered.load() >= 300; }));
    pipe.stop();
    REQUIRE(delivered.load() == 300);
    // 池化开启：容器复用发生（命中>0、归还>0）
    REQUIRE(pipe.pool_hits() > 0);
    REQUIRE(pipe.pool_returns() > 0);
}

TEST_CASE("pipeline 关闭池化：每帧新建、无命中", "[video][pipeline][pool]") {
    auto fake = std::make_shared<FakeDecoderBackend>();
    fake->frames_to_produce = 100;
    VideoDecoderConfig cfg;
    cfg.pooling = false;
    cfg.backpressure = Backpressure::Block;
    cfg.async_queue_size = 64;
    DecodePipeline pipe(fake, cfg);
    std::atomic<int> delivered{0};
    pipe.set_callback([&](VideoFrame&&) { delivered++; });
    std::string err;
    REQUIRE(pipe.open("fake://src", &err));
    REQUIRE(pipe.start(&err));
    REQUIRE(wait_until([&] { return delivered.load() >= 100; }));
    pipe.stop();
    REQUIRE(delivered.load() == 100);
    REQUIRE(pipe.pool_hits() == 0);
}

// === 背压 ===
TEST_CASE("pipeline 背压 Drop：队列满丢新帧、无死锁", "[video][pipeline][backpressure]") {
    auto fake = std::make_shared<FakeDecoderBackend>();
    fake->frames_to_produce = 100;
    VideoDecoderConfig cfg;
    cfg.backpressure = Backpressure::Drop;
    cfg.async_queue_size = 2;
    cfg.pooling = true;
    DecodePipeline pipe(fake, cfg);
    std::atomic<int> delivered{0};
    // 慢消费者：15ms/帧，decode 线程产帧更快 → 队列(2)满后持续丢帧
    pipe.set_callback([&](VideoFrame&&) {
        delivered++;
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    });
    std::string err;
    REQUIRE(pipe.open("fake://src", &err));
    REQUIRE(pipe.start(&err));
    REQUIRE(wait_until([&] { return fake->produced_count() >= 100; }));
    pipe.stop();  // 不得死锁
    REQUIRE(pipe.stats().dropped > 0);
    REQUIRE(delivered.load() < 100);  // 有帧被丢弃
}

TEST_CASE("pipeline 背压 OverwriteOldest：覆盖队首丢最旧帧", "[video][pipeline][backpressure]") {
    auto fake = std::make_shared<FakeDecoderBackend>();
    fake->frames_to_produce = 100;
    VideoDecoderConfig cfg;
    cfg.backpressure = Backpressure::OverwriteOldest;
    cfg.async_queue_size = 2;
    cfg.pooling = true;
    DecodePipeline pipe(fake, cfg);
    std::atomic<int> delivered{0};
    pipe.set_callback([&](VideoFrame&&) {
        delivered++;
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    });
    std::string err;
    REQUIRE(pipe.open("fake://src", &err));
    REQUIRE(pipe.start(&err));
    REQUIRE(wait_until([&] { return fake->produced_count() >= 100; }));
    pipe.stop();
    REQUIRE(pipe.stats().dropped > 0);
    // 覆盖最旧帧：被保留的是最新帧，部分被覆盖丢弃
    REQUIRE(delivered.load() > 0);
}

TEST_CASE("pipeline 背压 Block：队列不满时不丢帧", "[video][pipeline][backpressure]") {
    auto fake = std::make_shared<FakeDecoderBackend>();
    fake->frames_to_produce = 100;
    VideoDecoderConfig cfg;
    cfg.backpressure = Backpressure::Block;
    cfg.async_queue_size = 2;
    cfg.pooling = true;
    DecodePipeline pipe(fake, cfg);
    std::atomic<int> delivered{0};
    pipe.set_callback([&](VideoFrame&&) { delivered++; });
    std::string err;
    REQUIRE(pipe.open("fake://src", &err));
    REQUIRE(pipe.start(&err));
    REQUIRE(wait_until([&] { return delivered.load() >= 100; }));
    pipe.stop();
    REQUIRE(delivered.load() == 100);  // Block 策略下一帧不丢
    REQUIRE(pipe.stats().dropped == 0);
}

// === 重连 ===
TEST_CASE("pipeline 重连：瞬态失败按 delay 重试、成功收敛并计数", "[video][pipeline][reconnect]") {
    auto fake = std::make_shared<FakeDecoderBackend>();
    fake->frames_to_produce = 300;
    fake->transient_read_at = 50;      // 第 50 次 read 瞬态失败一次
    fake->reconnect_open_failures = 1; // 重连第 1 次 open 失败、第 2 次成功
    VideoDecoderConfig cfg;
    cfg.backpressure = Backpressure::Block;
    cfg.async_queue_size = 64;
    cfg.pooling = true;
    cfg.reconnect_delay_ms = 5;
    cfg.max_reconnects = 10;
    DecodePipeline pipe(fake, cfg);
    std::atomic<int> delivered{0};
    pipe.set_callback([&](VideoFrame&&) { delivered++; });
    std::string err;
    REQUIRE(pipe.open("fake://src", &err));
    REQUIRE(pipe.start(&err));
    REQUIRE(wait_until([&] { return delivered.load() >= 300; }));
    pipe.stop();
    REQUIRE(delivered.load() == 300);      // 重连后继续解出全部帧
    REQUIRE(pipe.stats().reconnect_count == 1);
}

TEST_CASE("pipeline EOF 不触发重连", "[video][pipeline][reconnect]") {
    auto fake = std::make_shared<FakeDecoderBackend>();
    fake->frames_to_produce = 100;
    fake->transient_read_at = -1;  // 无瞬态失败，正常 EOF
    VideoDecoderConfig cfg;
    cfg.backpressure = Backpressure::Block;
    cfg.async_queue_size = 32;
    cfg.pooling = true;
    cfg.reconnect_delay_ms = 5;
    cfg.max_reconnects = 10;
    DecodePipeline pipe(fake, cfg);
    std::atomic<int> delivered{0};
    pipe.set_callback([&](VideoFrame&&) { delivered++; });
    std::string err;
    REQUIRE(pipe.open("fake://src", &err));
    REQUIRE(pipe.start(&err));
    REQUIRE(wait_until([&] { return delivered.load() >= 100; }));
    pipe.stop();
    REQUIRE(pipe.stats().reconnect_count == 0);  // EOF 绝不重连
    REQUIRE(pipe.state() == State::Eof);
}

TEST_CASE("pipeline 重连超过 max_reconnects 报错", "[video][pipeline][reconnect]") {
    auto fake = std::make_shared<FakeDecoderBackend>();
    fake->frames_to_produce = 100;
    fake->transient_read_at = 1;      // 第一帧就读失败
    fake->reconnect_open_failures = 100;  // 重连永远失败
    VideoDecoderConfig cfg;
    cfg.backpressure = Backpressure::Block;
    cfg.async_queue_size = 8;
    cfg.pooling = true;
    cfg.reconnect_delay_ms = 1;
    cfg.max_reconnects = 3;
    DecodePipeline pipe(fake, cfg);
    std::string err;
    REQUIRE(pipe.open("fake://src", &err));
    REQUIRE(pipe.start(&err));
    // 等待重连尝试耗尽（初始 open + max_reconnects 次重连）
    REQUIRE(wait_until([&] { return fake->open_count() >= 1 + 3; }));
    pipe.stop();
    REQUIRE(pipe.state() == State::Error);           // 达上限报永久失败
    REQUIRE(pipe.stats().reconnect_count == 0);      // 从未成功
}

// === 双后端集成（真源）===
TEST_CASE("pipeline 异步 FFmpeg 解码 clip.h264（池化+背压）", "[video][pipeline][ffmpeg][integration]") {
    std::ifstream probe("test_data/video/clip.h264");
    if (!probe.good()) SKIP("no test clip; place at test_data/video/clip.h264");
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::FFmpeg;
    cfg.backpressure = Backpressure::Block;
    cfg.async_queue_size = 8;
    cfg.pooling = true;
    cfg.reconnect_delay_ms = 5;
    cfg.max_reconnects = 3;
    auto vd = VideoDecoder::create(cfg);
    REQUIRE(vd != nullptr);
    std::string err;
    REQUIRE(vd->open("test_data/video/clip.h264", &err));
    std::atomic<int> delivered{0};
    vd->set_callback([&](VideoFrame&& f) {
        REQUIRE_FALSE(f.image.empty());
        delivered++;
    });
    REQUIRE(vd->start(&err));
    bool done = wait_until(
        [&] { return vd->state() == State::Eof || vd->state() == State::Error; }, 120000);
    INFO("ffmpeg done=" << done << " state=" << state_to_string(vd->state())
                        << " delivered=" << delivered.load() << " last_err=" << vd->last_error());
    REQUIRE(done);
    vd->stop();
    REQUIRE(delivered.load() > 0);
    REQUIRE(vd->pool_hits() > 0);  // 真实后端池化命中
    vd->close();
}

TEST_CASE("pipeline 异步 GStreamer 解码 clip.h264（池化+背压）", "[video][pipeline][gst][integration]") {
    std::ifstream probe("test_data/video/clip.h264");
    if (!probe.good()) SKIP("no test clip; place at test_data/video/clip.h264");
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::GStreamer;
    cfg.backpressure = Backpressure::Block;
    cfg.async_queue_size = 8;
    cfg.pooling = true;
    cfg.reconnect_delay_ms = 5;
    cfg.max_reconnects = 3;
    auto vd = VideoDecoder::create(cfg);
    REQUIRE(vd != nullptr);
    std::string err;
    REQUIRE(vd->open("test_data/video/clip.h264", &err));
    std::atomic<int> delivered{0};
    vd->set_callback([&](VideoFrame&& f) {
        REQUIRE_FALSE(f.image.empty());
        delivered++;
    });
    REQUIRE(vd->start(&err));
    bool done = wait_until(
        [&] { return vd->state() == State::Eof || vd->state() == State::Error; }, 120000);
    INFO("gst done=" << done << " state=" << state_to_string(vd->state())
                     << " delivered=" << delivered.load() << " last_err=" << vd->last_error());
    REQUIRE(done);
    vd->stop();
    REQUIRE(delivered.load() > 0);
    REQUIRE(vd->pool_hits() > 0);
    vd->close();
}

