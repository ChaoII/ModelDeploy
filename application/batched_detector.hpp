#pragma once
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "config.hpp"
#include "csrc/pipeline/async_model.h"
#include "csrc/serving/adapters.h"
#include "csrc/vision/detection/ultralytics_det.h"
#include "csrc/vision/common/result.h"

// 共享批处理检测器：**一个模型实例 + 单 worker 攒批**（复用 SDK AsyncModel）。
// 多路 channel 共享同一实例 → 多帧凑 batch 送一次推理，GPU/CPU 利用率高、权重只驻留一份。
// 与「每路 clone 各自实例」相比：解码/编码仍每路独立，仅推理共用；这是 GPU 上做 10×24 的关键。
class BatchedDetector {
public:
    explicit BatchedDetector(const ModelConfig& cfg, size_t max_batch = 8,
                             std::chrono::milliseconds batch_timeout = std::chrono::milliseconds(3));
    ~BatchedDetector();

    BatchedDetector(const BatchedDetector&) = delete;
    BatchedDetector& operator=(const BatchedDetector&) = delete;

    bool ok() const { return ok_; }
    const std::string& error() const { return err_; }

    // 提交一帧并等待结果（内部走 AsyncModel，多路并发时自动攒批）
    bool predict(const modeldeploy::vision::ImageData& image,
                 std::vector<modeldeploy::vision::DetectionResult>* out);

    // 供 pipeline 绘制（只读标签 + 写图）；模型实例由本对象持有、存活期内有效
    modeldeploy::vision::detection::UltralyticsDet* model() const { return raw_; }
    std::unordered_map<int, std::string> label_map() const;

    uint64_t batch_runs() const { return async_ ? async_->batch_runs() : 0; }

private:
    using RM = modeldeploy::serving::ResultModel<modeldeploy::vision::detection::UltralyticsDet,
                                                 std::vector<modeldeploy::vision::DetectionResult>>;
    modeldeploy::vision::detection::UltralyticsDet* raw_ = nullptr;
    std::unique_ptr<modeldeploy::pipeline::AsyncModel<RM>> async_;
    bool ok_ = false;
    std::string err_;
};
