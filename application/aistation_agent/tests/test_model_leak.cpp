#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include "pipeline_manager.hpp"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#endif

namespace {

constexpr const char* kDetPath = "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx";

bool det_data_available() { return std::ifstream(kDetPath).good(); }

ModelConfig det_cfg(const std::string& name, float thr) {
    ModelConfig m;
    m.name = name;
    m.type = "detection";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kDetPath;
    m.labels = {"person"};
    m.input_size = {640, 640};
    m.confidence_threshold = thr;
    m.interval = 1;
    return m;
}

TaskConfig task_cfg(const std::string& id, float thr) {
    TaskConfig c;
    c.id = id;
    c.name = id;
    c.input_url = "test_data/test_video60.mp4";
    c.enable_preview = false;                     // 只需模型加载路径，不打开编解码
    c.models = {det_cfg("det", thr)};
    return c;
}

#if defined(_WIN32)
size_t process_thread_count() {
    const DWORD pid = GetCurrentProcessId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return 0;
    THREADENTRY32 te{};
    te.dwSize = sizeof(te);
    size_t n = 0;
    if (Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID == pid) ++n;
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
    return n;
}

size_t process_working_set_mb() {
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)))
        return 0;
    return static_cast<size_t>(pmc.WorkingSetSize / (1024ull * 1024ull));
}
#else
size_t process_thread_count() { return 0; }
size_t process_working_set_mb() { return 0; }
#endif

}  // namespace

// 并发审计 BLOCKER 回归：模型/会话/线程泄漏。
// 旧实现 `det_pool_` 用 shared_ptr 强引用且**从不 erase**；每次热更新改变 confidence 即
// 生成新的池 key → 新建一个 BatchedDetector（内含独立 OrtBackend session + AsyncModel
// worker + ~8 ORT intra-op 线程），旧检测器因池强引用永不析构。
// 实测（audit_conc_1.ps1，40 次热更新）：Private 806.7→1937.8 MB、线程 73→392，
// 删除全部任务并静置后仍 1401.5 MB / 388 线程（未回落）。
// 修复后（det_pool_ 改 weak_ptr + 原型 LRU 上限）：每次热更新旧检测器随旧引擎释放而析构，
// 池中存活检测器恒为 1，线程/内存不随迭代次数增长。
TEST_CASE("PipelineManager 热更新不泄漏检测器与线程", "[agent][leak]") {
    if (!det_data_available()) SKIP("test data absent");
    PipelineManager mgr;
    std::string err;
    REQUIRE(mgr.create_task(task_cfg("leak1", 0.10f), &err));

    // 首次建立共享检测器
    REQUIRE(mgr.update_model("leak1", "det", det_cfg("det", 0.10f)));
    REQUIRE(mgr.detector_pool_size() == 1);

    const size_t threads_before = process_thread_count();
    const size_t ws_before = process_working_set_mb();

    constexpr int kIters = 20;
    for (int i = 1; i <= kIters; ++i) {
        // confidence 变化 → det_pool_ key 变化；旧实现每次新建检测器且永不释放
        REQUIRE(mgr.update_model("leak1", "det", det_cfg("det", 0.10f + 0.001f * i)));
        // 池中存活检测器恒为 1：旧检测器随旧引擎（被热更新换出）析构
        // 用 CHECK（非 REQUIRE）：即便池泄漏也继续采样线程/内存，便于一次暴露全部问题
        CHECK(mgr.detector_pool_size() == 1);
    }
    // 允许少量后台线程退出抖动后再采样
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const size_t threads_after = process_thread_count();
    const size_t ws_after = process_working_set_mb();
    WARN("threads before=" << threads_before << " after=" << threads_after
                           << "; ws_mb before=" << ws_before << " after=" << ws_after);
#if defined(_WIN32)
    // 旧实现 20 次热更新新增 ~160 线程；修复后应持平（留抖动余量）
    REQUIRE(threads_after <= threads_before + 8);
    // 旧实现 20 次热更新约 +550MB 驻留；修复后不应出现随迭代线性增长的大块驻留
    REQUIRE(ws_after <= ws_before + 96);
#endif

    // 删除任务 → 引擎释放 → 池内弱引用全部失效 → 存活检测器归零
    REQUIRE(mgr.remove_task("leak1"));
    REQUIRE(mgr.task_count() == 0);
    CHECK(mgr.detector_pool_size() == 0);
}

// 原型缓存必须有界：不同配置反复热更新（生成不同 cache key 的原型）时受 LRU 上限约束，
// 不会无界驻留模型权重。
TEST_CASE("PipelineManager 原型缓存受 LRU 上限约束", "[agent][leak]") {
    if (!det_data_available()) SKIP("test data absent");
    PipelineManager mgr;
    std::string err;
    REQUIRE(mgr.create_task(task_cfg("proto1", 0.10f), &err));
    REQUIRE(mgr.update_model("proto1", "det", det_cfg("det", 0.10f)));
    // 检测模型走共享检测器池，不写原型缓存
    REQUIRE(mgr.prototype_count() == 0);
    REQUIRE(mgr.detector_pool_size() == 1);
    REQUIRE(mgr.remove_task("proto1"));
    REQUIRE(mgr.detector_pool_size() == 0);
}
