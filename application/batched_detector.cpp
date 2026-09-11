#include "batched_detector.hpp"

#include <algorithm>
#include <thread>

#include "runtime_factory.hpp"

using namespace modeldeploy;
using namespace modeldeploy::vision;

BatchedDetector::BatchedDetector(const ModelConfig& cfg, const size_t max_batch,
                                 const std::chrono::milliseconds batch_timeout) {
    RuntimeOption opt = build_runtime_option(cfg);
    // CPU 上让一个 batch 内多帧并行（intra-op 线程数 ≥ batch 并行度）；GPU 忽略此设置。
    if (cfg.device != "gpu" && cfg.device != "tpu") {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const int threads = static_cast<int>(std::max<size_t>(1, std::min<size_t>(hw, max_batch)));
        opt.set_cpu_thread_num(threads);
    }
    auto model = std::make_unique<RM>(cfg.path, opt);
    if (!model->is_initialized()) {
        err_ = "failed to initialize detection model: " + cfg.path;
        return;
    }
    if (cfg.input_size.size() == 2) model->get_preprocessor().set_size(cfg.input_size);
    model->get_postprocessor().set_conf_threshold(cfg.confidence_threshold);
    raw_ = model.get();

    modeldeploy::pipeline::AsyncModelConfig acfg;
    acfg.max_batch = max_batch;
    acfg.batch_timeout = batch_timeout;
    acfg.enable_batching = true;
    async_ = std::make_unique<modeldeploy::pipeline::AsyncModel<RM>>(std::move(model), acfg);
    std::string serr;
    if (!async_->start(&serr)) {
        err_ = "AsyncModel start failed: " + serr;
        raw_ = nullptr;
        async_.reset();
        return;
    }
    ok_ = true;
}

BatchedDetector::~BatchedDetector() {
    if (async_) async_->stop();
}

bool BatchedDetector::predict(const modeldeploy::vision::ImageData& image,
                              std::vector<modeldeploy::vision::DetectionResult>* out) {
    if (!ok_ || !out || !async_) return false;
    try {
        auto fut = async_->predict_async(image);
        *out = fut.get();
        return true;
    } catch (...) {
        return false;
    }
}

std::unordered_map<int, std::string> BatchedDetector::label_map() const {
    if (!raw_) return {};
    try {
        return raw_->get_label_map("names");
    } catch (...) {
        return {};
    }
}
