#pragma once
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "config.hpp"
#include "inference_engine.hpp"
#include "perf_stats.hpp"
#include "frame_pool.hpp"
#include "csrc/vision/common/image_data.h"

/// 多模型调度组：管理同一路视频上的多个推理模型
/// 每模型一个常驻 worker 线程，避免 std::async 反复创建线程
class InferGroup {
public:
    using ModelFactory = std::function<std::unique_ptr<InferenceEngine>(const ModelConfig&)>;

    explicit InferGroup(const TaskConfig& cfg, ModelFactory factory = nullptr);
    ~InferGroup();

    /// 初始化所有模型 + 启动 worker 线程池
    bool init();

    /// 对一帧执行所有模型的推理（多模型并行）— BGR 路径
    /// @param y_device/uv_device CUVID 硬解 GPU NV12 指针（GPU 直通零拷贝，可为 null）
    /// @param frame_out 输出的 BGR 图（供绘制/快照；need_bgr=false 时可为空）
    /// @param need_bgr 是否需要生成 BGR 图（非预览路可跳过，GPU 直通时省一次 GPU 往返）
    /// @return 0=无模型需要处理（全部跳帧），>0=有模型实际执行了推理
    int run_models(uint8_t* y_plane, uint8_t* uv_plane,
                    const uint8_t* y_device, const uint8_t* uv_device,
                    int width, int height, int y_step, int uv_step,
                    std::vector<InferResult>* results,
                    modeldeploy::vision::ImageData* frame_out = nullptr,
                    bool need_bgr = true);

    /// 检查是否所有模型都支持 CUDA NV12 预处理（用于 GPU 快速路径判断）
    bool all_cuda_preproc() const;

    PerfStats& stats() { return stats_; }
    bool ready() const;

    /// 是否可走 GPU NV12 直通（所有模型为 detection + device=gpu + 无 ROI）
    bool gpu_nv12_ready() const { return gpu_nv12_ready_; }

    bool add_model(const ModelConfig& cfg);
    bool remove_model(const std::string& name);
    bool update_model(const std::string& name, const ModelConfig& cfg);

private:
    TaskConfig cfg_;
    ModelFactory factory_;
    // 模型列表互斥：run_models 与 add/remove/update_model 串行化，防止遍历期间 engines_ 被改写
    std::mutex models_mtx_;
    std::vector<std::unique_ptr<InferenceEngine>> engines_;
    std::vector<int> frame_counters_;
    // GPU NV12 直通可用（detection + gpu + 无 ROI）：推理跳过 host BGR 转换
    bool gpu_nv12_ready_ = false;
    PerfStats stats_;
    FramePool frame_pool_;
    std::atomic<bool> initialized_{false};

    // 复用缓冲：避免每帧分配
    std::vector<uint8_t> nv12_buf_;
    std::vector<uint8_t> bgr_buf_;
    int last_w_ = 0, last_h_ = 0;

    // 每个模型一个常驻 worker（避免 std::async 重复创建线程）
    struct Worker {
        std::thread thread;
        std::mutex mtx;
        std::condition_variable cv_in;
        std::condition_variable cv_out;
        std::function<void()> task;
        bool has_task = false;
        bool done = true;
        bool stop = false;
    };
    std::vector<std::unique_ptr<Worker>> workers_;

    void worker_loop(Worker* w);
    void start_workers();
    void stop_workers();

    // 后台 warm-up 线程：TRT 首次编译可耗时数十秒，移到后台避免阻塞解码/停止
    std::thread warmup_thread_;
    void start_warmup();
};
