#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <functional>

#include "config.hpp"
#include "perf_stats.hpp"
#include "infer_group.hpp"
#include "draw_engine.hpp"
#include "video_source.hpp"
#include "video_sink.hpp"

namespace modeldeploy { namespace vision { class ImageData; } }

/// 单路视频流水线：SDK 解码(async)→有界队列→单检测线程(推理+绘制+编码)→SDK 编码(async)
/// 解码为 SDK 异步 + 缓冲池；检测循环为每帧串行关键路径（detect+draw+encode < 40ms 以达 25fps）。
/// 背压由有界队列在满时丢最旧帧实现（保最新、控延迟）。
class Pipeline {
public:
    using ModelFactory = std::function<std::unique_ptr<InferenceEngine>(const ModelConfig&)>;

    explicit Pipeline(TaskConfig cfg, ModelFactory factory = nullptr);
    ~Pipeline();

    bool start();
    void stop();
    bool is_running() const { return running_.load(); }
    bool is_initialized() const { return initialized_.load(); }
    // 线程安全：多线程读写 init_error_（detect/HTTP）
    std::string init_error() const;
    void set_init_error(const std::string& msg);

    const std::string& task_id() const { return cfg_.id; }
    const PerfStats& stats() const { return stats_; }
    const TaskConfig& config() const { return cfg_; }

    bool add_model(const ModelConfig& mcfg);
    bool remove_model(const std::string& name);
    bool update_model(const std::string& name, const ModelConfig& mcfg);

    /// 更新任务核心配置（仅支持 enable_preview、encoder、decoder、draw 字段；需停止任务后调用）
    bool update_config(const TaskConfig& cfg);
    void update_preview_mode(bool enable);

    /// 获取最新一帧 BGR JPEG（线程安全）
    bool latest_jpeg(std::vector<uint8_t>* out, int quality = 80);

    /// 仅取最新帧快照（shared_ptr 拷贝，快；可在全局锁内调用）
    bool latest_bgr_snapshot(std::shared_ptr<modeldeploy::vision::ImageData>* out) const;

    /// 对快照编码 JPEG（无锁，可在全局锁外调用；5-15ms 级）
    static bool encode_jpeg(const std::shared_ptr<modeldeploy::vision::ImageData>& snap,
                            std::vector<uint8_t>* out, int quality);

    /// 是否启用预览编码
    bool is_preview_enabled() const { return cfg_.enable_preview; }

private:
    TaskConfig cfg_;
    ModelFactory model_factory_;
    InferGroup infer_group_;
    std::unique_ptr<DrawEngine> draw_engine_;   // 非 detection（face/classification）CPU 标注
    VideoSource src_;
    VideoSink sink_;
    PerfStats stats_;

    std::atomic<bool> initialized_{false};
    mutable std::mutex init_error_mtx_;
    std::string init_error_;

    // 最新一帧（用于 HTTP 快照）
    mutable std::mutex snapshot_mtx_;
    std::shared_ptr<modeldeploy::vision::ImageData> latest_bgr_;
    int snapshot_interval_ = 2;

    // 最新检测结果缓存（保留成员：供跳帧复用绘制的扩展）
    std::vector<InferResult> cached_results_;

    // 源流帧率（从解码器自动检测）
    int source_fps_ = 25;

    std::atomic<bool> running_{false};
    std::atomic<bool> stopped_{true};       // 防止重复 stop
    // start/stop 生命周期互斥：防止 stop 在线程创建完成前 join（未 join 的 thread 析构会 std::terminate）
    std::mutex lifecycle_mtx_;
    std::thread detect_thread_;

    // 生命周期令牌：解码回调持有其强引用；release_resources 时置 false，
    // 保证任务析构后解码线程的在途回调不再触碰 this（防 use-after-free）
    std::shared_ptr<std::atomic<bool>> alive_token_;

    // ── 解码回调 → 检测循环 的有界队列（满时丢最旧帧保帧率） ──
    std::deque<modeldeploy::video::VideoFrame> det_queue_;
    std::mutex det_mtx_;
    std::condition_variable det_cv_;
    size_t det_max_size_ = 3;

    // 最近处理帧的毫秒时间戳（SDR 解码帧）
    std::atomic<int64_t> last_frame_pts_{0};

    /// 检测循环（应用单线程关键路径）
    void detect_loop();

    /// 解码回调向有界队列推帧
    void push_detect_frame(modeldeploy::video::VideoFrame&& f);

    /// 低频快照生成（对设备帧回读 CPU；不占 25fps 关键路径）
    void update_snapshot(const modeldeploy::vision::ImageData& frame, int64_t& counter);

    /// 按模型名取置信度阈值（缺省 0.5）
    double model_threshold(const std::string& name) const;

    /// 非 detection（face/classification）结果标注到帧（DrawEngine；NV12 就地/回环绘制）
    void draw_non_det(modeldeploy::vision::ImageData& frame,
                      const std::vector<InferResult>& results);

    /// 资源安全释放（仅在所有线程结束后调用）
    void release_resources();
};
