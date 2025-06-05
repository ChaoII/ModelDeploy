//
// Created by aichao on 2025/6/5.
//

#pragma once

#include <chrono>
#include <vector>
#include <array>
#include "tabulate/tabulate.hpp"

class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;

    void start() {
        start_time_ = Clock::now();
    }

    void stop() {
        end_time_ = Clock::now();
        durations_.push_back(std::chrono::duration<double, std::milli>(end_time_ - start_time_).count());
    }

    [[nodiscard]] double average_ms() const {
        if (durations_.empty()) return 0.0;
        double sum = 0.0;
        for (const double d : durations_) sum += d;
        return sum / durations_.size();
    }

    void reset() {
        durations_.clear();
    }

    void print(const std::string& tag) const {
        std::cout << termcolor::cyan << tag << ": avg = " << average_ms() << " ms" << termcolor::reset << std::endl;
    }

    void set_durations(const std::vector<double>& durations) {
        durations_ = durations;
    }

    [[nodiscard]] std::vector<double> get_durations() const {
        return durations_;
    }


    Timer operator+(const Timer& other) const {
        Timer result;
        const size_t n = std::min(durations_.size(), other.durations_.size());
        result.durations_.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            result.durations_.push_back(durations_[i] + other.durations_[i]);
        }
        return result;
    }

    Timer& operator+=(const Timer& other) {
        const size_t n = std::min(durations_.size(), other.durations_.size());
        for (size_t i = 0; i < n; ++i) {
            durations_[i] += other.durations_[i];
        }
        return *this;
    }

private:
    Clock::time_point start_time_;
    Clock::time_point end_time_;
    std::vector<double> durations_;
};

struct TimerArray {
    Timer pre_timer;
    Timer infer_timer;
    Timer post_timer;

    [[nodiscard]] double total_ms() const {
        return pre_timer.average_ms() + infer_timer.average_ms() + post_timer.average_ms();
    }

    TimerArray operator+(const TimerArray& other) const {
        return TimerArray{
            pre_timer + other.pre_timer,
            infer_timer + other.infer_timer,
            post_timer + other.post_timer
        };
    }

    // 重载 +=
    TimerArray& operator+=(const TimerArray& other) {
        pre_timer += other.pre_timer;
        infer_timer += other.infer_timer;
        post_timer += other.post_timer;
        return *this;
    }


    void reset() {
        pre_timer.reset();
        infer_timer.reset();
        post_timer.reset();
    }

    void print_benchmark() const {
        pre_timer.print("[Preprocess ]");
        infer_timer.print("[Inference  ]");
        post_timer.print("[Postprocess]");
        std::cout << termcolor::magenta << "[Total      ]" << ": avg = "
            << total_ms() << " ms" << termcolor::reset << std::endl;
    }
};
