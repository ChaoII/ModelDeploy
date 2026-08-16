//
// Created by aichao on 2025/6/5.
//

#include "utils/benchmark.h"
#include <tabulate/tabulate.hpp>



    void Timer::start() {
        start_time_ = Clock::now();
    }

    void Timer::stop() {
        end_time_ = Clock::now();
        durations_.push_back(std::chrono::duration<double, std::milli>(end_time_ - start_time_).count());
    }

    // 平均耗时：单次 predict = 该次耗时；多次 start/stop = 平均每次
    [[nodiscard]] double Timer::average_ms() const {
        if (durations_.empty()) return 0.0;
        double sum = 0.0;
        for (const double d : durations_) sum += d;
        return sum / durations_.size();
    }

    // 总耗时：所有计时段之和（pipeline 内多次子模型共用时 = 累计）
    [[nodiscard]] double Timer::total_ms() const {
        double sum = 0.0;
        for (const double d : durations_) sum += d;
        return sum;
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





    // 单帧平均总耗时（= 各阶段单次平均之和）。
    // 注意：demo 循环多次 predict 复用同一 TimerArray 时，average_ms 是单次平均，
    // total_ms() 也应返回单次平均，与 print_benchmark 的 pre/infer/post 一致。
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

