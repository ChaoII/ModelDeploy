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

    double average_ms() const {
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

private:
    Clock::time_point start_time_;
    Clock::time_point end_time_;
    std::vector<double> durations_;
};

struct TimerArray {
    Timer pre_timer;
    Timer infer_timer;
    Timer post_timer;

    double total_ms() const {
        return pre_timer.average_ms() + infer_timer.average_ms() + post_timer.average_ms();
    }

    void print_benchmark() const {
        pre_timer.print("[Preprocess ]");
        infer_timer.print("[Inference  ]");
        post_timer.print("[Postprocess]");
        std::cout << termcolor::magenta << "[Total      ]" << ": avg = "
            << total_ms() << " ms" << termcolor::reset << std::endl;
    }
};
