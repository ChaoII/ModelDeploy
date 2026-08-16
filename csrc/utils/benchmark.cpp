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

    double Timer::mean_ms() const {
        if (durations_.empty()) return 0.0;
        double sum = 0.0;
        for (const double d : durations_) sum += d;
        return sum / durations_.size();
    }

    double Timer::sum_ms() const {
        double sum = 0.0;
        for (const double d : durations_) sum += d;
        return sum;
    }

    size_t Timer::count() const {
        return durations_.size();
    }

    void Timer::add_sample(const double duration_ms) {
        durations_.push_back(duration_ms);
    }

    void Timer::reset() {
        durations_.clear();
    }

    // 逐样本相加：较短的补 0 对齐，避免 std::min 静默截断丢数据
    Timer Timer::operator+(const Timer& other) const {
        Timer result;
        const size_t n = durations_.size() > other.durations_.size() ? durations_.size() : other.durations_.size();
        result.durations_.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const double a = i < durations_.size() ? durations_[i] : 0.0;
            const double b = i < other.durations_.size() ? other.durations_[i] : 0.0;
            result.durations_.push_back(a + b);
        }
        return result;
    }

    Timer& Timer::operator+=(const Timer& other) {
        const size_t n = durations_.size() > other.durations_.size() ? durations_.size() : other.durations_.size();
        durations_.resize(n, 0.0);
        for (size_t i = 0; i < other.durations_.size(); ++i) {
            durations_[i] += other.durations_[i];
        }
        return *this;
    }

    double TimerArray::mean_ms() const {
        return pre_timer.mean_ms() + infer_timer.mean_ms() + post_timer.mean_ms();
    }

    double TimerArray::sum_ms() const {
        return pre_timer.sum_ms() + infer_timer.sum_ms() + post_timer.sum_ms();
    }

    TimerArray TimerArray::operator+(const TimerArray& other) const {
        return TimerArray{
            pre_timer + other.pre_timer,
            infer_timer + other.infer_timer,
            post_timer + other.post_timer
        };
    }

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
        // 单帧平均（各段 mean + 总 mean），与 demo 单帧循环语义一致
        const double pre = pre_timer.mean_ms();
        const double infer = infer_timer.mean_ms();
        const double post = post_timer.mean_ms();
        std::cout << termcolor::cyan << "[Preprocess ]: avg = " << pre << " ms" << termcolor::reset << std::endl;
        std::cout << termcolor::cyan << "[Inference  ]: avg = " << infer << " ms" << termcolor::reset << std::endl;
        std::cout << termcolor::cyan << "[Postprocess]: avg = " << post << " ms" << termcolor::reset << std::endl;
        std::cout << termcolor::magenta << "[Total      ]: avg = "
            << (pre + infer + post) << " ms" << termcolor::reset << std::endl;
    }
