#include "perf_stats.hpp"
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>

void PerfStats::start() {
    start_time = std::chrono::steady_clock::now();
}

void PerfStats::record_frame(int64_t dec, int64_t inf, int64_t drw, int64_t enc) {
    std::lock_guard<std::mutex> lock(mtx);
    decode_us.push_back(dec);
    infer_us.push_back(inf);
    draw_us.push_back(drw);
    encode_us.push_back(enc);
    total_us.push_back(dec + inf + drw + enc);
    ++frame_count;
}

void PerfStats::reset() {
    std::lock_guard<std::mutex> lock(mtx);
    decode_us.clear();
    infer_us.clear();
    draw_us.clear();
    encode_us.clear();
    total_us.clear();
    frame_count = 0;
    start_time = std::chrono::steady_clock::now();
}

static double avg_vec(const std::vector<int64_t>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size() / 1000.0;
}

double PerfStats::avg_decode_ms() const { std::lock_guard<std::mutex> lock(mtx); return avg_vec(decode_us); }
double PerfStats::avg_infer_ms() const { std::lock_guard<std::mutex> lock(mtx); return avg_vec(infer_us); }
double PerfStats::avg_draw_ms() const { std::lock_guard<std::mutex> lock(mtx); return avg_vec(draw_us); }
double PerfStats::avg_encode_ms() const { std::lock_guard<std::mutex> lock(mtx); return avg_vec(encode_us); }
double PerfStats::avg_total_ms() const { std::lock_guard<std::mutex> lock(mtx); return avg_vec(total_us); }

double PerfStats::fps() const {
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    return (elapsed > 0 && frame_count > 0) ? frame_count / elapsed : 0.0;
}

int64_t PerfStats::elapsed_sec() const {
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
    printf("  FPS                 : %.1f\n", fps());
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
       << "\"avg_total_ms\":" << avg_total_ms()
       << "}";
    return os.str();
}
