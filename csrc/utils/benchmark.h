//
// Created by aichao on 2025/6/5.
//

#pragma once

#include <chrono>
#include <vector>
#include "core/md_decl.h"

class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;

    void start();

    void stop();

    [[nodiscard]] double average_ms() const;

    void push_back(const double duration);

    void reset();

    void print(const std::string& tag) const;

    void set_durations(const std::vector<double>& durations);

    [[nodiscard]] std::vector<double> get_durations() const;


    Timer operator+(const Timer& other) const;

    Timer& operator+=(const Timer& other);

private:
    Clock::time_point start_time_;
    Clock::time_point end_time_;
    std::vector<double> durations_;
};

struct MODELDEPLOY_CXX_EXPORT TimerArray {
    Timer pre_timer;
    Timer infer_timer;
    Timer post_timer;

    [[nodiscard]] double total_ms() const;

    TimerArray operator+(const TimerArray& other) const;

    // 重载 +=
    TimerArray& operator+=(const TimerArray& other);

    void reset();

    void print_benchmark() const;
};
