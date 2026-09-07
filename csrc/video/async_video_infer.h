//
// AsyncVideoInfer —— 解码→异步推理薄适配器
//
// 把解码器逐帧回调（FrameCallback：`void(VideoFrame&&)`）直接接线到 AsyncModel 的异步推理：
// 每帧取内嵌 ImageData（`frame.image`，零拷贝）投递到 AsyncModel，不做任何转换。
// 本适配器**不拥有** AsyncModel，生命周期由调用方管理（配合 AsyncModel::stop() 后再析构）。
// 解码器通常单线程调 on_frame；AsyncModel::predict_async_cb 在多 producer 并发下亦安全。
//

#pragma once

#include <atomic>
#include <utility>

#include "pipeline/async_model.h"
#include "video/video_frame.h"

namespace modeldeploy::video {

template <typename M>
class AsyncVideoInfer {
public:
    using Callback = typename modeldeploy::pipeline::AsyncModel<M>::Callback;

    explicit AsyncVideoInfer(modeldeploy::pipeline::AsyncModel<M>& infer) : infer_(infer) {}

    // 供解码器接线的回调：转发 frame.image 到 infer_.predict_async_cb
    void on_frame(modeldeploy::video::VideoFrame&& frame) {
        infer_.predict_async_cb(frame.image);
        ++frames_submitted_;
    }

    // 推理完成回调（转投 AsyncModel::set_callback）
    void set_result_callback(Callback cb) { infer_.set_callback(std::move(cb)); }

    // 已投递帧数
    uint64_t frames_submitted() const { return frames_submitted_.load(); }

private:
    modeldeploy::pipeline::AsyncModel<M>& infer_;
    std::atomic<uint64_t> frames_submitted_{0};
};

} // namespace modeldeploy::video
