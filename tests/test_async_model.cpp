#include <catch2/catch_all.hpp>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
#include "vision/common/image_data.h"
#include "utils/benchmark.h"
#include "pipeline/async_model.h"
#include "video/async_video_infer.h"
#include "video/video_frame.h"

namespace {
// 仿 UltralyticsDet 的同步接口；R = std::string
struct FakeModel {
    using result_type = std::string;
    bool ok = true;
    std::chrono::milliseconds delay{0};
    std::atomic<size_t> predict_calls{0};
    std::atomic<size_t> batch_calls{0};

    static std::string big_result(int w, int h) {
        return "w=" + std::to_string(w) + ",h=" + std::to_string(h);
    }
    bool predict(const modeldeploy::vision::ImageData& img, std::string* out,
                 TimerArray* /*timers*/ = nullptr) {
        ++predict_calls;
        if (delay.count() > 0) std::this_thread::sleep_for(delay);
        if (!ok) return false;
        *out = big_result(img.width(), img.height());
        return true;
    }
    bool batch_predict(const std::vector<modeldeploy::vision::ImageData>& imgs,
                       std::vector<std::string>* outs,
                       TimerArray* /*timers*/ = nullptr) {
        ++batch_calls;
        if (delay.count() > 0) std::this_thread::sleep_for(delay);
        outs->clear();
        if (!ok) return false;
        for (auto& im : imgs) outs->push_back(big_result(im.width(), im.height()));
        return true;
    }
};

// from_bgr24(nullptr,...) 会经 from_raw 拒绝空指针返回空图，故用栈缓冲 + from_raw 构造最小有效图。
modeldeploy::vision::ImageData make_bgr(int w, int h) {
    std::vector<uint8_t> buf(static_cast<size_t>(w) * h * 3);
    return modeldeploy::vision::ImageData::from_raw(buf.data(), w, h,
                                                    MdImageType::PKG_BGR_U8,
                                                    false, modeldeploy::Device::CPU);
}
} // namespace

TEST_CASE("FakeModel batch_predict aggregates results", "[async]") {
    FakeModel m;
    std::vector<modeldeploy::vision::ImageData> imgs = {make_bgr(4, 2), make_bgr(8, 3)};
    std::vector<std::string> outs;
    REQUIRE(m.batch_predict(imgs, &outs));
    REQUIRE(outs.size() == 2);
    REQUIRE(outs[0] == "w=4,h=2");
    REQUIRE(outs[1] == "w=8,h=3");
}

using namespace modeldeploy::pipeline;

TEST_CASE("AsyncModel future non-blocking single prediction", "[async]") {
    auto model = std::make_unique<FakeModel>();
    AsyncModelConfig cfg;
    cfg.enable_batching = false;
    AsyncModel<FakeModel> am(std::move(model), cfg);
    REQUIRE(am.start());
    auto fut = am.predict_async(make_bgr(16, 9));
    REQUIRE(am.submitted() >= 1);
    REQUIRE(fut.get() == FakeModel::big_result(16, 9));
    am.wait_idle();
    REQUIRE(am.pending() == 0);
    am.stop();
    REQUIRE(am.completed() == am.submitted());
}

TEST_CASE("AsyncModel batches multiple predictions", "[async]") {
    auto model = std::make_unique<FakeModel>();
    FakeModel* m = model.get();
    AsyncModelConfig cfg;
    cfg.enable_batching = true;
    cfg.max_batch = 8;
    AsyncModel<FakeModel> am(std::move(model), cfg);
    REQUIRE(am.start());
    std::vector<std::future<std::string>> futs;
    for (int i = 1; i <= 16; ++i) futs.push_back(am.predict_async(make_bgr(i, i + 1)));
    am.wait_idle();
    REQUIRE(am.batch_runs() >= 2);
    REQUIRE(m->batch_calls > 0);
    for (int i = 1; i <= 16; ++i)
        REQUIRE(futs[static_cast<size_t>(i - 1)].get() == FakeModel::big_result(i, i + 1));
    REQUIRE(am.pending() == 0);
    am.stop();
}

TEST_CASE("AsyncModel backpressure does not drop tasks", "[async]") {
    auto model = std::make_unique<FakeModel>();
    model->delay = std::chrono::milliseconds(30);
    AsyncModelConfig cfg;
    cfg.queue_capacity = 2;
    AsyncModel<FakeModel> am(std::move(model), cfg);
    REQUIRE(am.start());
    const int kCount = 50;
    std::vector<std::future<std::string>> futs;
    for (int i = 0; i < kCount; ++i) futs.push_back(am.predict_async(make_bgr(3 + i, 5)));
    for (int i = 0; i < kCount; ++i)
        REQUIRE(futs[static_cast<size_t>(i)].get() == FakeModel::big_result(3 + i, 5));
    REQUIRE(am.completed() == am.submitted());
    REQUIRE(am.pending() == 0);
    am.stop();
}

TEST_CASE("AsyncModel callback matches future results", "[async]") {
    auto model = std::make_unique<FakeModel>();
    AsyncModelConfig cfg;
    cfg.enable_batching = true;
    cfg.max_batch = 4;
    AsyncModel<FakeModel> am(std::move(model), cfg);
    std::vector<std::pair<uint64_t, std::string>> cb;
    std::mutex cb_mu;
    am.set_callback([&](uint64_t id, std::string&& r, const std::string& err) {
        std::lock_guard<std::mutex> lk(cb_mu);
        if (err.empty()) cb.emplace_back(id, std::move(r));
    });
    REQUIRE(am.start());
    std::vector<uint64_t> ids;
    for (int i = 0; i < 5; ++i) ids.push_back(am.predict_async_cb(make_bgr(2 + i, 3)));
    am.wait_idle();
    REQUIRE(am.completed() == am.submitted());
    for (size_t i = 1; i < ids.size(); ++i) REQUIRE(ids[i] > ids[i - 1]);
    std::lock_guard<std::mutex> lk(cb_mu);
    REQUIRE(cb.size() == ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        REQUIRE(cb[i].first == ids[i]);
        REQUIRE(cb[i].second == FakeModel::big_result(static_cast<int>(2 + i), 3));
    }
    am.stop();
}

TEST_CASE("AsyncModel surfaces model errors via future and callback", "[async]") {
    auto model = std::make_unique<FakeModel>();
    model->ok = false;
    AsyncModelConfig cfg;
    cfg.enable_batching = true;
    AsyncModel<FakeModel> am(std::move(model), cfg);
    std::vector<std::string> cb_errors;
    std::mutex cb_mu;
    am.set_callback([&](uint64_t, std::string&&, const std::string& err) {
        std::lock_guard<std::mutex> lk(cb_mu);
        cb_errors.push_back(err);
    });
    REQUIRE(am.start());
    auto fut = am.predict_async(make_bgr(4, 4));
    bool threw = false;
    try {
        fut.get();
    } catch (const std::exception&) {
        threw = true;
    }
    REQUIRE(threw);
    am.wait_idle();
    REQUIRE(am.completed() == am.submitted());
    std::lock_guard<std::mutex> lk(cb_mu);
    REQUIRE(cb_errors.size() >= 1);
    for (auto& e : cb_errors) REQUIRE(!e.empty());
    am.stop();
}

TEST_CASE("AsyncVideoInfer wires decode frames to async inference", "[async]") {
    auto m = std::make_unique<FakeModel>();
    AsyncModelConfig cfg;
    AsyncModel<FakeModel> am(std::move(m), cfg);
    REQUIRE(am.start());
    modeldeploy::video::AsyncVideoInfer<FakeModel> avi(am);
    std::vector<std::string> got;
    std::mutex got_mu;
    avi.set_result_callback([&](uint64_t /*id*/, std::string&& r, const std::string& e) {
        if (e.empty()) {
            std::lock_guard<std::mutex> lk(got_mu);
            got.push_back(std::move(r));
        }
    });
    modeldeploy::video::VideoFrame f1;
    f1.image = make_bgr(4, 2);
    avi.on_frame(std::move(f1));
    modeldeploy::video::VideoFrame f2;
    f2.image = make_bgr(8, 3);
    avi.on_frame(std::move(f2));
    am.wait_idle();
    REQUIRE(avi.frames_submitted() == 2);
    {
        std::lock_guard<std::mutex> lk(got_mu);
        REQUIRE(got.size() == 2);
        REQUIRE(got[0] == FakeModel::big_result(4, 2));
        REQUIRE(got[1] == FakeModel::big_result(8, 3));
    }
    am.stop();
}

TEST_CASE("AsyncVideoInfer accumulates many frames without blocking", "[async]") {
    auto m = std::make_unique<FakeModel>();
    AsyncModelConfig cfg;
    AsyncModel<FakeModel> am(std::move(m), cfg);
    REQUIRE(am.start());
    modeldeploy::video::AsyncVideoInfer<FakeModel> avi(am);
    std::atomic<size_t> got{0};
    avi.set_result_callback([&](uint64_t /*id*/, std::string&&, const std::string& e) {
        if (e.empty()) ++got;
    });
    const int kN = 100;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kN; ++i) {
        modeldeploy::video::VideoFrame f;
        f.image = make_bgr(1 + i, 2);
        avi.on_frame(std::move(f));
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    REQUIRE(avi.frames_submitted() == static_cast<uint64_t>(kN));
    REQUIRE(elapsed < 2000);   // on_frame 不阻塞
    am.wait_idle();
    REQUIRE(got.load() == static_cast<size_t>(kN));
    avi.set_result_callback({});   // 复位回调防后续污染
    am.stop();
}

TEST_CASE("AsyncModel stop is idempotent and wait_idle drains", "[async]") {
    auto model = std::make_unique<FakeModel>();
    model->delay = std::chrono::milliseconds(5);
    AsyncModelConfig cfg;
    cfg.queue_capacity = 4;
    AsyncModel<FakeModel> am(std::move(model), cfg);
    REQUIRE(am.start());
    for (int i = 0; i < 10; ++i) am.predict_async(make_bgr(4, 4));
    am.wait_idle();
    REQUIRE(am.pending() == 0);
    REQUIRE(am.completed() == am.submitted());
    am.stop();
    am.stop();   // 幂等：第二次不抛异常
    REQUIRE(am.pending() == 0);
}
