//
// AsyncModel —— 流式/异步推理通用模板
//
// 并发正确性铁律：BaseModel 复用成员缓冲（reused_input_tensors_/reused_output_tensors_），
// 同一模型实例**绝不**可并发调用 predict/batch_predict。故 AsyncModel 内部**只有单一推理
// 线程**负责调用模型方法；吞吐靠 batch 合并 + 多实例（多个 AsyncModel 对象），而非对单实例
// 并发上锁。num_workers 字段保留但本期恒按 1（>1 需 M::clone 克隆多实例，本期不实现）。
//

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "vision/common/image_data.h"
#include "utils/benchmark.h"

namespace modeldeploy::pipeline {

using vision::ImageData;

struct AsyncModelConfig {
    size_t queue_capacity = 256;
    size_t num_workers = 1;          // 本期恒按 1（单推理线程）；>1 预留（需 M::clone，本期不实现）
    size_t max_batch = 8;
    std::chrono::milliseconds batch_timeout{2};
    bool enable_batching = true;
};

template <typename M>
class AsyncModel {
public:
    using ResultType = typename M::result_type;   // 若 M 无 result_type，编译期即报错
    using Callback = std::function<void(uint64_t task_id, ResultType&& result, const std::string& error)>;

    AsyncModel(std::unique_ptr<M> model, const AsyncModelConfig& cfg = {})
        : model_(std::move(model)), cfg_(cfg) {}
    ~AsyncModel() { stop(); }

    AsyncModel(const AsyncModel&) = delete;
    AsyncModel& operator=(const AsyncModel&) = delete;
    AsyncModel(AsyncModel&&) = delete;
    AsyncModel& operator=(AsyncModel&&) = delete;

    bool start(std::string* err = nullptr) {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        if (started_ || stopped_) {
            if (err) *err = "AsyncModel already started or stopped";
            return false;
        }
        started_ = true;
        worker_ = std::thread(&AsyncModel::worker_loop, this);
        return true;
    }

    // 幂等：置停止 → 唤醒 → 对队列内 promise 注错 → join
    void stop() {
        {
            std::lock_guard<std::mutex> lk(queue_mutex_);
            if (stopped_) return;               // 幂等
            stopped_ = true;
            while (!queue_.empty()) {
                auto t = std::move(queue_.front());
                queue_.pop_front();
                fail(t.promise, "AsyncModel stopped");
                ++completed_;
                --pending_;
            }
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    std::future<ResultType> predict_async(const ImageData& image, std::string* err = nullptr) {
        auto promise = std::make_shared<std::promise<ResultType>>();
        std::future<ResultType> fut = promise->get_future();
        submit(image, std::move(promise), err);
        return fut;
    }

    void set_callback(Callback cb) {   // start 前设置；可重复设置（原子替换）
        std::lock_guard<std::mutex> lk(callback_mutex_);
        callback_ = std::move(cb);
    }

    uint64_t predict_async_cb(const ImageData& image, std::string* err = nullptr) {
        // 内部用默认 promise 不等待；仅承担 completed/pending 计数 + 保证 id 递增
        auto promise = std::make_shared<std::promise<ResultType>>();
        return submit(image, std::move(promise), err);
    }

    bool busy() const { return pending() > 0; }
    uint64_t pending() const { return pending_.load(); }
    uint64_t submitted() const { return submitted_.load(); }
    uint64_t completed() const { return completed_.load(); }
    uint64_t batch_runs() const { return batch_runs_.load(); }

    void wait_idle() {
        while (pending_.load() != 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

private:
    struct Task {
        uint64_t id;
        ImageData image;
        std::shared_ptr<std::promise<ResultType>> promise;
    };

    uint64_t submit(const ImageData& image, std::shared_ptr<std::promise<ResultType>> promise,
                    std::string* err) {
        std::unique_lock<std::mutex> lk(queue_mutex_);
        uint64_t id = next_id_++;
        cv_.wait(lk, [this] { return stopped_ || queue_.size() < cfg_.queue_capacity; });
        if (stopped_) {
            fail(promise, "AsyncModel stopped");
            if (err) *err = "AsyncModel stopped";
            return id;
        }
        queue_.push_back(Task{id, image, std::move(promise)});
        ++submitted_;
        ++pending_;
        cv_.notify_all();
        return id;
    }

    void worker_loop() {
        while (true) {
            std::vector<Task> batch = take_batch();
            if (batch.empty()) break;   // stopped
            dispatch(batch);
        }
    }

    // 攒一批最多 max_batch 个任务返回；单个时若 enable_batching 仍走 batch_predict（保持一致性）
    std::vector<Task> take_batch() {
        std::unique_lock<std::mutex> lk(queue_mutex_);
        cv_.wait(lk, [this] { return stopped_ || !queue_.empty(); });
        if (stopped_) return {};

        std::vector<Task> out;
        out.push_back(std::move(queue_.front()));
        queue_.pop_front();

        if (cfg_.enable_batching && cfg_.max_batch > 1) {
            auto deadline = std::chrono::steady_clock::now() + cfg_.batch_timeout;
            while (out.size() < cfg_.max_batch) {
                if (!queue_.empty()) {
                    out.push_back(std::move(queue_.front()));
                    queue_.pop_front();
                    continue;
                }
                if (!cv_.wait_until(lk, deadline, [this] { return stopped_ || !queue_.empty(); }))
                    break;   // 攒满或超时
                if (stopped_) break;
                out.push_back(std::move(queue_.front()));
                queue_.pop_front();
            }
        }
        cv_.notify_all();   // 释放空位唤醒可能的 producer
        return out;
    }

    void dispatch(std::vector<Task>& batch) {
        if (batch.empty()) return;
        TimerArray timers;

        if (cfg_.enable_batching) {
            ++batch_runs_;
            std::vector<ImageData> images;
            images.reserve(batch.size());
            for (auto& t : batch) images.push_back(t.image);

            std::vector<ResultType> results;
            std::string err;
            bool ok = false;
            try {
                ok = model_->batch_predict(images, &results, &timers);
            } catch (const std::exception& e) {
                err = e.what();
            } catch (...) {
                err = "unknown exception in batch_predict";
            }
            if (ok && results.size() == batch.size()) {
                for (size_t i = 0; i < batch.size(); ++i)
                    complete(batch[i], std::move(results[i]), "");
            } else {
                if (err.empty()) err = "batch_predict failed";
                for (auto& t : batch) complete(t, ResultType{}, err);
            }
        } else {
            for (auto& t : batch) {
                ResultType result;
                std::string err;
                bool ok = false;
                try {
                    ok = model_->predict(t.image, &result, &timers);
                } catch (const std::exception& e) {
                    err = e.what();
                } catch (...) {
                    err = "unknown exception in predict";
                }
                if (ok)
                    complete(t, std::move(result), "");
                else
                    complete(t, ResultType{}, err.empty() ? "predict failed" : err);
            }
        }
    }

    void complete(Task& t, ResultType&& result, const std::string& error) {
        Callback cb;
        {
            std::lock_guard<std::mutex> lk(callback_mutex_);
            cb = callback_;
        }
        if (error.empty()) {
            if (t.promise && cb) {
                ResultType copy = result;                       // future 与回调各自需要值
                t.promise->set_value(std::move(result));
                cb(t.id, std::move(copy), "");
            } else if (t.promise) {
                t.promise->set_value(std::move(result));
            } else if (cb) {
                cb(t.id, std::move(result), "");
            }
        } else {
            if (t.promise) t.promise->set_exception(std::make_exception_ptr(std::runtime_error(error)));
            if (cb) cb(t.id, ResultType{}, error);
        }
        ++completed_;
        --pending_;
    }

    static void fail(const std::shared_ptr<std::promise<ResultType>>& p, const std::string& msg) {
        if (p) p->set_exception(std::make_exception_ptr(std::runtime_error(msg)));
    }

    std::unique_ptr<M> model_;
    AsyncModelConfig cfg_;

    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::deque<Task> queue_;

    std::mutex callback_mutex_;
    Callback callback_;

    std::atomic<uint64_t> submitted_{0};
    std::atomic<uint64_t> completed_{0};
    std::atomic<uint64_t> batch_runs_{0};
    std::atomic<uint64_t> pending_{0};
    uint64_t next_id_ = 0;               // 在 queue_mutex_ 下访问

    std::thread worker_;
    bool started_ = false;               // queue_mutex_ 保护
    bool stopped_ = false;               // queue_mutex_ 保护
};

} // namespace modeldeploy::pipeline
