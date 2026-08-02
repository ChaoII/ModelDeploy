#include "infer_group.hpp"
#include "csrc/vision/common/image_data.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#ifdef WITH_GPU
#include "csrc/vision/common/processors/nv12_to_bgr.cuh"
#endif

using namespace modeldeploy::vision;

InferGroup::InferGroup(const TaskConfig& cfg, ModelFactory factory)
    : cfg_(cfg), factory_(factory), frame_pool_(32) {
}

InferGroup::~InferGroup() {
    stop_workers();
    if (warmup_thread_.joinable()) warmup_thread_.join();
    engines_.clear();
}

static std::unique_ptr<InferenceEngine> make_engine(
    const ModelConfig& mcfg, InferGroup::ModelFactory& factory) {
    if (factory) {
        auto eng = factory(mcfg);
        if (eng && eng->is_loaded()) return eng;
    }
    auto eng = std::make_unique<InferenceEngine>();
    if (eng->load(mcfg)) return eng;
    return nullptr;
}

bool InferGroup::init() {
    for (const auto& mcfg : cfg_.models) {
        auto engine = make_engine(mcfg, factory_);
        if (!engine) {
            std::cerr << "[InferGroup] Failed to init model: " << mcfg.name << std::endl;
            return false;
        }
        engines_.push_back(std::move(engine));
        frame_counters_.push_back(0);
    }
    if (engines_.empty()) return false;
    start_workers();

    // Warm-up 移入后台线程：TRT 首次编译可耗时 30-60s，若在解码线程同步执行，
    // stop() join 解码线程会阻塞数十秒（任务停止/删除时 HTTP 全卡）。
    std::cout << "[InferGroup] Warming up " << engines_.size() << " model(s) in background..."
              << std::endl;
    start_warmup();

    initialized_ = true;
    return true;
}

void InferGroup::start_warmup() {
    if (warmup_thread_.joinable()) return;
    warmup_thread_ = std::thread([this]() {
        cudaSetDevice(0);
        std::lock_guard<std::mutex> lock(models_mtx_);
        for (size_t i = 0; i < engines_.size(); ++i) {
            const auto& engine = engines_[i];
            const auto& mcfg = engine->config();
            int w = mcfg.input_size.size() == 2 ? mcfg.input_size[0] : 640;
            int h = mcfg.input_size.size() == 2 ? mcfg.input_size[1] : 640;
            auto dummy = ImageData::from_raw(std::vector<uint8_t>(h * w * 3, 0).data(),
                                              w, h, MdImageType::PKG_BGR_U8, true);
            InferResult dummy_result;
            auto t0 = std::chrono::steady_clock::now();
            engine->infer(dummy, &dummy_result);
            auto t1 = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::cout << "[InferGroup] Warm-up done: " << mcfg.name << " (" << ms << "ms)" << std::endl;
        }
    });
}

bool InferGroup::ready() const {
    return initialized_.load();
}

bool InferGroup::all_cuda_preproc() const {
    for (const auto& eng : engines_) {
        const auto& type = eng->config().type;
        if (type == "detection") {
            // Detection models have use_cuda_preproc via UltralyticsPreprocessor
            // Checked indirectly: if device == "gpu" and cuda_preproc was set
            if (eng->config().device != "gpu") return false;
        } else if (type == "face_detection") {
            if (eng->config().device != "gpu") return false;
        } else {
            // Unknown model type — can't guarantee CUDA preproc
            return false;
        }
    }
    return !engines_.empty();
}

void InferGroup::worker_loop(Worker* w) {
    cudaSetDevice(0); // 与主线程同 CUDA context
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(w->mtx);
            w->cv_in.wait(lock, [w]() { return w->has_task || w->stop; });
            if (w->stop && !w->has_task) return;
            task = std::move(w->task);
            w->has_task = false;
        }
        try { task(); } catch (...) {}
        {
            std::lock_guard<std::mutex> lock(w->mtx);
            w->done = true;
        }
        w->cv_out.notify_one();
    }
}

void InferGroup::start_workers() {
    workers_.clear();
    workers_.reserve(engines_.size());
    for (size_t i = 0; i < engines_.size(); ++i) {
        auto w = std::make_unique<Worker>();
        Worker* wp = w.get();
        w->thread = std::thread([this, wp]() { this->worker_loop(wp); });
        workers_.push_back(std::move(w));
    }
}

void InferGroup::stop_workers() {
    for (auto& w : workers_) {
        {
            std::lock_guard<std::mutex> lock(w->mtx);
            w->stop = true;
        }
        w->cv_in.notify_one();
    }
    for (auto& w : workers_) {
        if (w->thread.joinable()) w->thread.join();
    }
    workers_.clear();
}

int InferGroup::run_models(uint8_t* y_plane, uint8_t* uv_plane,
                              int width, int height, int y_step, int uv_step,
                              std::vector<InferResult>* results,
                              ImageData* frame_out) {
    if (!initialized_) return 0;
    // 全程持模型锁：与 add/remove/update_model 串行化，防止遍历 engines_/workers_ 时被改写
    std::lock_guard<std::mutex> lock(models_mtx_);
    results->clear();

    const size_t y_size = static_cast<size_t>(height) * width;
    const size_t uv_size = y_size / 2;
    const size_t total = y_size + uv_size;

    // 紧凑连续 NV12 直接使用（零拷贝），仅在非紧凑行距时才落到本地缓冲
    const uint8_t* y_src = y_plane;
    const uint8_t* uv_src = uv_plane;
    if (!(y_step == width && uv_step == width && y_plane && uv_plane)) {
        if (last_w_ != width || last_h_ != height || nv12_buf_.size() != total) {
            nv12_buf_.resize(total);
            last_w_ = width;
            last_h_ = height;
        }
        std::memcpy(nv12_buf_.data(), y_plane, y_size);
        std::memcpy(nv12_buf_.data() + y_size, uv_plane, uv_size);
        y_src = nv12_buf_.data();
        uv_src = nv12_buf_.data() + y_size;
    }

    // 预先判断是否有任一模型需要处理本帧（计数器 + 间隔检查）
    // 使用临时变量记录，避免下面 dispatch 循环重复 increment
    std::vector<bool> need_process(engines_.size(), false);
    bool any_needs_process = false;
    for (size_t i = 0; i < engines_.size(); ++i) {
        bool should = (++frame_counters_[i]) % engines_[i]->config().interval == 0;
        need_process[i] = should;
        if (should) any_needs_process = true;
    }

    // 即使所有模型跳推理，仍需输出 BGR 图像（供预览路保持 25fps 编码 + 复用缓存绘制）
#ifdef WITH_GPU
    const size_t bgr_size = static_cast<size_t>(height) * width * 3;
    if (bgr_buf_.size() < bgr_size) bgr_buf_.resize(bgr_size);
    nv12_to_bgr_cuda(y_src, uv_src,
                      width, height, width, width,
                      bgr_buf_.data());
    auto bgr_image = ImageData::from_raw(bgr_buf_.data(), width, height,
                                          MdImageType::PKG_BGR_U8, true); // copy=true：独立所有权，防跨队列缓冲别名竞争
#else
    auto nv12_image = ImageData::from_raw(y_src, width, height, MdImageType::NV12, true);
    auto bgr_image = ImageData::cvt_color(nv12_image, ColorConvertType::CVT_NV122PA_BGR);
#endif

    if (frame_out)
        *frame_out = bgr_image;

    // 所有模型跳推理 → 输出 BGR 但不输出结果
    if (!any_needs_process) {
        return 0;
    }

    struct ModelTask {
        size_t index;
        ImageData input;
        InferResult result;
        int64_t dt_us = 0;
        bool used = false;
    };
    std::vector<ModelTask> tasks(engines_.size());

    for (size_t i = 0; i < engines_.size(); ++i) {
        if (!need_process[i]) continue;
        auto& engine = engines_[i];
        const auto& mcfg = engine->config();
        ImageData input_image = bgr_image;
        if (mcfg.roi[2] > 0 && mcfg.roi[3] > 0) {
            Rect2f roi_rect = {static_cast<float>(mcfg.roi[0]),
                               static_cast<float>(mcfg.roi[1]),
                               static_cast<float>(mcfg.roi[2]),
                               static_cast<float>(mcfg.roi[3])};
            input_image = bgr_image.crop(roi_rect);
            if (input_image.empty()) continue;
        }
        tasks[i].index = i;
        tasks[i].input = std::move(input_image);
        tasks[i].used = true;
    }

    for (size_t i = 0; i < tasks.size(); ++i) {
        if (!tasks[i].used) continue;
        auto& w = workers_[i];
        auto& engine = engines_[i];
        ModelTask* tp = &tasks[i];
        {
            std::lock_guard<std::mutex> lock(w->mtx);
            w->done = false;
            w->has_task = true;
            w->task = [&engine, tp]() {
                auto t0 = std::chrono::steady_clock::now();
                engine->infer(tp->input, &tp->result);
                tp->dt_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count();
            };
        }
        w->cv_in.notify_one();
    }

    for (size_t i = 0; i < tasks.size(); ++i) {
        if (!tasks[i].used) continue;
        auto& w = workers_[i];
        std::unique_lock<std::mutex> lock(w->mtx);
        w->cv_out.wait(lock, [&w]() { return w->done; });
    }

    int ran_count = 0;
    for (auto& task : tasks) {
        if (!task.used) continue;
        ++ran_count;
        if (!task.result.boxes.empty())
            results->push_back(std::move(task.result));
        stats_.record_frame(0, task.dt_us, 0, 0);
    }

    return ran_count;
}

bool InferGroup::add_model(const ModelConfig& mcfg) {
    std::lock_guard<std::mutex> lock(models_mtx_);
    auto engine = std::make_unique<InferenceEngine>();
    if (!engine->load(mcfg)) return false;
    // 重启 worker 池以匹配新模型数量
    stop_workers();
    engines_.push_back(std::move(engine));
    frame_counters_.push_back(0);
    start_workers();
    return true;
}

bool InferGroup::remove_model(const std::string& name) {
    std::lock_guard<std::mutex> lock(models_mtx_);
    for (size_t i = 0; i < engines_.size(); ++i) {
        if (engines_[i]->config().name == name) {
            stop_workers();
            engines_.erase(engines_.begin() + i);
            frame_counters_.erase(frame_counters_.begin() + i);
            start_workers();
            return true;
        }
    }
    return false;
}

bool InferGroup::update_model(const std::string& name, const ModelConfig& mcfg) {
    std::lock_guard<std::mutex> lock(models_mtx_);
    for (auto& eng : engines_) {
        if (eng->config().name == name) {
            stop_workers();
            eng->unload();
            bool ok = eng->load(mcfg);
            start_workers();
            return ok;
        }
    }
    return false;
}
