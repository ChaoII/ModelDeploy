#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <map>
#include <deque>
#include <cstdint>

struct PerfStats {
    // 滑动窗口（最近 N 帧），避免长期运行后 FPS 失真 / 内存暴涨
    static constexpr size_t kWindow = 120;

    std::deque<int64_t> decode_us;
    std::deque<int64_t> infer_us;
    std::deque<int64_t> draw_us;
    std::deque<int64_t> encode_us;
    std::deque<int64_t> total_us;
    // 每帧的时间戳（用于 FPS 实时计算）
    std::deque<std::chrono::steady_clock::time_point> stamps;

    int64_t frame_count = 0;
    std::chrono::steady_clock::time_point start_time;
    mutable std::mutex mtx;

    // ── SDK 编解码统计摄入（来自 VideoSource::stats() 解码侧 / VideoSink::stats() 编码侧） ──
    // 与上面基于自测时延的 avg_* 语义不同，故使用独立 sdk_* 字段（向后兼容、纯增量）。
    uint64_t sdk_frames_in = 0;
    uint64_t sdk_frames_out = 0;
    uint64_t sdk_dropped = 0;
    uint64_t sdk_reconnect_count = 0;
    double sdk_avg_decode_ms = 0.0;
    double sdk_avg_encode_ms = 0.0;

    void start();
    void record_frame(int64_t dec, int64_t inf, int64_t drw, int64_t enc);
    /// 摄入 SDK 编解码 VideoStats（轻量：仅拷贝若干标量，可在 25fps 关键路径调用）
    void ingest_sdk(uint64_t frames_in, uint64_t frames_out, uint64_t dropped,
                    double avg_decode_ms, uint64_t reconnect_count, double avg_encode_ms);
    void reset();

    [[nodiscard]] double avg_decode_ms() const;
    [[nodiscard]] double avg_infer_ms() const;
    [[nodiscard]] double avg_draw_ms() const;
    [[nodiscard]] double avg_encode_ms() const;
    [[nodiscard]] double avg_total_ms() const;
    [[nodiscard]] double fps() const;
    [[nodiscard]] int64_t elapsed_sec() const;

    void print() const;
    std::string to_json() const;
};
