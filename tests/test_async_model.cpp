#include <catch2/catch_all.hpp>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
#include "vision/common/image_data.h"
#include "utils/benchmark.h"

namespace {
// 仿 UltralyticsDet 的同步接口；R = std::string
struct FakeModel {
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
