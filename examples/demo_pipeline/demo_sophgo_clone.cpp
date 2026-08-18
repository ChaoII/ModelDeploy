//
// 严格验证 Sophgo TPU 上模型 Clone 真实共享 TPU 存储（不重载 bmodel、不复制权重）。
//
// 内存法不可靠：libsophon bmrt_load_bmodel 对同一 bmodel 文件会去重复用，gmem heap
// mem_used 不统计权重区，内存增量无法区分共享/重载。改用无歧义的最硬指标——耗时：
//   真共享 clone()：不读磁盘、不重建 TPU engine → 微秒级
//   假实现 clone()：重读 bmodel、重建 engine → 毫秒级（≈一次完整加载）
// 断言 avg(clone) << avg(完整加载)。
//
// 阶段A（耗时）：N 次 clone() vs 1 次完整加载（双 bmodel），断言 clone 快几个数量级。
// 阶段B（并发）：原模型 + N 个 clone 多线程并发 predict，全部得到相同属性数量。
// 阶段C（生命周期）：original 在块内析构后，逃逸的 clone 仍能 predict 出相同结果。

#include "csrc/vision.h"
#include "bmlib_runtime.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace modeldeploy;
using namespace modeldeploy::vision;

using clk = std::chrono::steady_clock;
static double ms(clk::time_point t0, clk::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static void configure(pipeline::PedestrianAttribute& pa) {
    std::vector<int> det_size = {640, 640};
    const auto det_shape = pa.get_detector()->get_input_info(0).shape;
    if (det_shape.size() >= 4 && det_shape[2] > 0 && det_shape[3] > 0) {
        det_size = {static_cast<int>(det_shape[3]), static_cast<int>(det_shape[2])};
    }
    std::vector<int> cls_size = {192, 256};
    const auto cls_shape = pa.get_classifier()->get_input_info(0).shape;
    if (cls_shape.size() >= 4 && cls_shape[2] > 0 && cls_shape[3] > 0) {
        cls_size = {static_cast<int>(cls_shape[3]), static_cast<int>(cls_shape[2])};
    }
    pa.set_det_input_size(det_size);
    pa.set_det_threshold(0.5f);
    pa.set_cls_input_size(cls_size);
    pa.set_cls_batch_size(1);   // Sophgo int8 bmodel 是 batch=1 静态形状
}

int main(int argc, char** argv) {
    const std::string det_model = argc > 1 ? argv[1]
        : "../../test_data/test_models/sophgo/zhgd_without_nms_640_int8.bmodel";
    const std::string ml_model = argc > 2 ? argv[2]
        : "../../test_data/test_models/sophgo/zhgd_ml_int8.bmodel";
    const std::string image_path = argc > 3 ? argv[3]
        : "../../test_data/test_images/test_pedestrian_attribute_scale.png";
    const int nclones = argc > 4 ? atoi(argv[4]) : 4;

    RuntimeOption option;
    option.use_sophgo_backend(0);
    printf("[backend] Sophgo TPU: det=%s ml=%s clones=%d\n",
           det_model.c_str(), ml_model.c_str(), nclones);

    auto img = ImageData::imread(image_path);
    if (img.empty()) { printf("failed to read image\n"); return 1; }

    size_t expect = 0;
    double load_ms = 0.0, clone_ms = 0.0;
    std::vector<std::unique_ptr<pipeline::PedestrianAttribute>> clones;

    // ===== 块内：original 加载 + clone；块结束 original 析构 =====
    {
        auto t0 = clk::now();
        pipeline::PedestrianAttribute original(det_model, ml_model, option);
        if (!original.is_initialized()) { printf("original init failed\n"); return 1; }
        configure(original);
        auto t1 = clk::now();
        load_ms = ms(t0, t1);

        std::vector<AttributeResult> r_orig;
        original.predict(img, &r_orig);
        expect = r_orig.size();
        printf("full load (2 bmodels):   %9.3f ms    original attributes=%zu\n",
               load_ms, expect);

        auto tc0 = clk::now();
        for (int i = 0; i < nclones; ++i) {
            clones.emplace_back(original.clone());
            if (!clones.back() || !clones.back()->is_initialized()) {
                printf("clone[%d] failed\n", i);
                return 1;
            }
        }
        auto tc1 = clk::now();
        clone_ms = ms(tc0, tc1);
    }
    // ---- original 已析构；clones 逃逸存活 ----

    // ===== 阶段A：耗时 =====
    const double avg_clone = clone_ms / nclones;
    const bool instant = (load_ms > 1.0) && (avg_clone < load_ms / 100.0);
    printf("%d x clone():              %9.3f ms  (avg %8.4f ms)\n",
           nclones, clone_ms, avg_clone);
    printf("[A-TIMING] avg clone %.4f ms vs full load %.3f ms => INSTANT_CLONE=%s\n",
           avg_clone, load_ms, instant ? "YES" : "NO");

    // ===== 阶段B：并发 —— original 已销毁，仅并发跑 N 个 clone =====
    std::vector<size_t> counts(nclones, 0);
    std::vector<std::thread> threads;
    for (int i = 0; i < nclones; ++i) {
        threads.emplace_back([&, i] {
            std::vector<AttributeResult> r;
            if (clones[i]->predict(img, &r)) counts[i] = r.size();
        });
    }
    for (auto& th : threads) th.join();
    bool concurrent_ok = true;
    for (size_t c : counts) if (c != expect) concurrent_ok = false;
    printf("[B-CONCURRENT] %d clones parallel predict => OK=%s\n",
           nclones, concurrent_ok ? "YES" : "NO");

    // ===== 阶段C：生命周期 —— original 已析构，clone 仍可用 =====
    std::vector<AttributeResult> r_confirm;
    clones[0]->predict(img, &r_confirm);
    const bool lifetime_ok = (r_confirm.size() == expect);
    printf("[C-LIFETIME] after original destroyed, clone predicts %zu/%zu => OK=%s\n",
           r_confirm.size(), expect, lifetime_ok ? "YES" : "NO");

    const bool ok = instant && concurrent_ok && lifetime_ok;
    printf("\n[VERDICT] INSTANT=%s CONCURRENT=%s LIFETIME=%s => %s\n",
           instant ? "YES" : "NO", concurrent_ok ? "YES" : "NO",
           lifetime_ok ? "YES" : "NO",
           ok ? "SOPHGO CLONE REAL SHARED TPU - PASS" : "SOPHGO CLONE SHARING FAILED");
    return ok ? 0 : 2;
}
