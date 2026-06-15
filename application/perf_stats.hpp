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

    void start();
    void record_frame(int64_t dec, int64_t inf, int64_t drw, int64_t enc);
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
