//
// Created by aichao on 2025/6/5.
//
// 计时工具：Timer（单段采样）+ TimerArray（pre/infer/post 三段）。
//
// 架构约定：
//  - Timer 记录多次 start/stop 的时长样本；mean_ms()/sum_ms()/count() 为只读查询。
//  - TimerArray 的三个 Timer 语义一致（同一帧三段各采样一次），
//    mean_ms() = 单帧平均总耗时；sum_ms() = 多次/多子模型累积总耗时。
//  - demo/examples 打印单帧平均用 print_benchmark()（mean）；
//    benchmark/测试聚合多轮用 sum_ms()。
//

#pragma once

#include <chrono>
#include <vector>
#include <string>
#include "core/md_decl.h"

class MODELDEPLOY_CXX_EXPORT Timer {
public:
    using Clock = std::chrono::high_resolution_clock;

    void start();
    void stop();

    /// 多次采样平均耗时（ms）。无样本返回 0。
    [[nodiscard]] double mean_ms() const;
    /// 所有采样总耗时（ms）。pipeline 内多子模型共用同一 Timer 时 = 累积耗时。
    [[nodiscard]] double sum_ms() const;
    /// 采样次数。
    [[nodiscard]] size_t count() const;

    /// 手动追加一个时长样本（单位 ms）。用于 pipeline 层对 pre/post 段占位。
    void add_sample(double duration_ms);

    void reset();

    /// 逐样本相加（对齐补 0，避免静默截断）。
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

    /// 单帧平均总耗时（ms）= 三段 mean 之和。与 print_benchmark 输出一致。
    [[nodiscard]] double mean_ms() const;
    /// 累积总耗时（ms）= 三段 sum 之和。多轮/multi 子模型聚合用。
    [[nodiscard]] double sum_ms() const;

    TimerArray operator+(const TimerArray& other) const;
    TimerArray& operator+=(const TimerArray& other);

    void reset();

    /// 打印单帧平均耗时（各段 mean + 总 mean）。
    void print_benchmark() const;
};
