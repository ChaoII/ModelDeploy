//
// Created by aichao on 2025/6/5.
//

#pragma once

#include <chrono>
#include <vector>
#include <array>
#include "utils/benchmark.h"
#include <tabulate/tabulate.hpp>



    void Timer::start() {
        start_time_ = Clock::now();
    }

    void Timer::stop() {
        end_time_ = Clock::now();
        durations_.push_back(std::chrono::duration<double, std::milli>(end_time_ - start_time_).count());
    }

    [[nodiscard]] double Timer::average_ms() const {
        if (durations_.empty()) return 0.0;
        double sum = 0.0;
        for (const double d : durations_) sum += d;
        return sum / durations_.size();
    }

    void Timer::push_back(const double duration) {
        durations_.push_back(duration);
    }

    void Timer::reset() {
        durations_.clear();
    }

    void Timer::print(const std::string& tag) const {
        std::cout << termcolor::cyan << tag << ": avg = " << average_ms() << " ms" << termcolor::reset << std::endl;
    }

    void Timer::set_durations(const std::vector<double>& durations) {
        durations_ = durations;
    }

    [[nodiscard]] std::vector<double> Timer::get_durations() const {
        return durations_;
    }


    Timer Timer::operator+(const Timer& other) const {
        Timer result;
        const size_t n = std::min(durations_.size(), other.durations_.size());
        result.durations_.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            result.durations_.push_back(durations_[i] + other.durations_[i]);
        }
        return result;
    }

    Timer& Timer::operator+=(const Timer& other) {
        const size_t n = std::min(durations_.size(), other.durations_.size());
        for (size_t i = 0; i < n; ++i) {
            durations_[i] += other.durations_[i];
        }
        return *this;
    }





    [[nodiscard]] double TimerArray::total_ms() const {
        return pre_timer.average_ms() + infer_timer.average_ms() + post_timer.average_ms();
    }

    TimerArray TimerArray::operator+(const TimerArray& other) const {
        return TimerArray{
            pre_timer + other.pre_timer,
            infer_timer + other.infer_timer,
            post_timer + other.post_timer
        };
    }

    // 重载 +=
    TimerArray& TimerArray::operator+=(const TimerArray& other) {
        pre_timer += other.pre_timer;
        infer_timer += other.infer_timer;
        post_timer += other.post_timer;
        return *this;
    }


    void TimerArray::reset() {
        pre_timer.reset();
        infer_timer.reset();
        post_timer.reset();
    }

    void TimerArray::print_benchmark() const {
        pre_timer.print("[Preprocess ]");
        infer_timer.print("[Inference  ]");
        post_timer.print("[Postprocess]");
        std::cout << termcolor::magenta << "[Total      ]" << ": avg = "
            << total_ms() << " ms" << termcolor::reset << std::endl;
    }

