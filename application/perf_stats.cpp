#include "perf_stats.hpp"
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>

void PerfStats::start() {
    std::lock_guard<std::mutex> lock(mtx);
    start_time = std::chrono::steady_clock::now();
    decode_us.clear();
    infer_us.clear();
    draw_us.clear();
    encode_us.clear();
    total_us.clear();
    stamps.clear();
    frame_count = 0;
    sdk_frames_in = 0;
    sdk_frames_out = 0;
    sdk_dropped = 0;
    sdk_reconnect_count = 0;
    sdk_avg_decode_ms = 0.0;
    sdk_avg_encode_ms = 0.0;
}

static inline void push_window(std::deque<int64_t>& q, int64_t v) {
    q.push_back(v);
    if (q.size() > PerfStats::kWindow) q.pop_front();
}

void PerfStats::record_frame(int64_t dec, int64_t inf, int64_t drw, int64_t enc) {
    std::lock_guard<std::mutex> lock(mtx);
    push_window(decode_us, dec);
    push_window(infer_us, inf);
    push_window(draw_us, drw);
    push_window(encode_us, enc);
    push_window(total_us, dec + inf + drw + enc);
    stamps.push_back(std::chrono::steady_clock::now());
    if (stamps.size() > kWindow) stamps.pop_front();
    ++frame_count;
}

void PerfStats::ingest_sdk(uint64_t frames_in, uint64_t frames_out, uint64_t dropped,
                           double avg_decode_ms, uint64_t reconnect_count, double avg_encode_ms) {
    sdk_frames_in = frames_in;
    sdk_frames_out = frames_out;
    sdk_dropped = dropped;
    sdk_reconnect_count = reconnect_count;
    sdk_avg_decode_ms = avg_decode_ms;
    sdk_avg_encode_ms = avg_encode_ms;
}

void PerfStats::reset() {
    std::lock_guard<std::mutex> lock(mtx);
    decode_us.clear();
    infer_us.clear();
    draw_us.clear();
    encode_us.clear();
    total_us.clear();
    stamps.clear();
    frame_count = 0;
    start_time = std::chrono::steady_clock::now();
    sdk_frames_in = 0;
    sdk_frames_out = 0;
    sdk_dropped = 0;
    sdk_reconnect_count = 0;
    sdk_avg_decode_ms = 0.0;
    sdk_avg_encode_ms = 0.0;
}

static double avg_us_to_ms(const std::deque<int64_t>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size() / 1000.0;
}

double PerfStats::avg_decode_ms() const { std::lock_guard<std::mutex> lock(mtx); return avg_us_to_ms(decode_us); }
double PerfStats::avg_infer_ms()  const { std::lock_guard<std::mutex> lock(mtx); return avg_us_to_ms(infer_us); }
double PerfStats::avg_draw_ms()   const { std::lock_guard<std::mutex> lock(mtx); return avg_us_to_ms(draw_us); }
double PerfStats::avg_encode_ms() const { std::lock_guard<std::mutex> lock(mtx); return avg_us_to_ms(encode_us); }
double PerfStats::avg_total_ms()  const { std::lock_guard<std::mutex> lock(mtx); return avg_us_to_ms(total_us); }

// 用滑动窗口内最早帧到最新帧的时间差计算瞬时 FPS（更真实）
double PerfStats::fps() const {
    std::lock_guard<std::mutex> lock(mtx);
    if (stamps.size() < 2) return 0.0;
    auto dt = std::chrono::duration<double>(stamps.back() - stamps.front()).count();
    if (dt <= 0) return 0.0;
    return static_cast<double>(stamps.size() - 1) / dt;
}

int64_t PerfStats::elapsed_sec() const {
    std::lock_guard<std::mutex> lock(mtx);
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time).count();
}

void PerfStats::print() const {
    auto f = [](const std::string& n, double v) {
        printf("  %-18s: %7.2f ms\n", n.c_str(), v);
    };
    printf("── PerfStats ─────────────────────\n");
    printf("  Frames              : %lld\n", (long long)frame_count);
    printf("  Elapsed             : %lld s\n", (long long)elapsed_sec());
    printf("  FPS (window)        : %.1f\n", fps());
    f("Decode (avg)", avg_decode_ms());
    f("Infer (avg)", avg_infer_ms());
    f("Draw (avg)", avg_draw_ms());
    f("Encode (avg)", avg_encode_ms());
    f("Pipeline total (avg)", avg_total_ms());
    printf("──────────────────────────────────\n");
}

std::string PerfStats::to_json() const {
    std::ostringstream os;
    os << "{"
       << "\"frames\":" << frame_count << ","
       << "\"elapsed_sec\":" << elapsed_sec() << ","
       << "\"fps\":" << fps() << ","
       << "\"avg_decode_ms\":" << avg_decode_ms() << ","
       << "\"avg_infer_ms\":" << avg_infer_ms() << ","
       << "\"avg_draw_ms\":" << avg_draw_ms() << ","
       << "\"avg_encode_ms\":" << avg_encode_ms() << ","
       << "\"avg_total_ms\":" << avg_total_ms() << ","
       // SDK 编解码统计（Task 7 摄入；纯增量，不影响上游既有字段）
       << "\"sdk_frames_in\":" << sdk_frames_in << ","
       << "\"sdk_frames_out\":" << sdk_frames_out << ","
       << "\"sdk_dropped\":" << sdk_dropped << ","
       << "\"sdk_reconnect_count\":" << sdk_reconnect_count << ","
       << "\"sdk_avg_decode_ms\":" << sdk_avg_decode_ms << ","
       << "\"sdk_avg_encode_ms\":" << sdk_avg_encode_ms
       << "}";
    return os.str();
}
