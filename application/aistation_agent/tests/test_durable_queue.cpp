#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "event_publisher.hpp"

namespace fs = std::filesystem;

namespace {
struct FakeTransport : EventPublisher {
    std::atomic<bool> online{true};
    std::vector<std::string> got;
    std::mutex m;
    bool publish(const DetectionEvent& e) override {
        if (!online.load()) return false;
        std::lock_guard<std::mutex> lk(m);
        got.push_back(e.event_id);
        return true;
    }
    std::string name() const override { return "fake"; }
};

DetectionEvent ev(const std::string& id) {
    DetectionEvent e;
    e.event_id = id; e.edge_code = "edge-01"; e.camera_id = 1; e.task_id = 1;
    e.ts = "2026-09-12T08:00:00.000Z";
    return e;
}
}  // namespace

TEST_CASE("DurableQueue flushes in order when online", "[agent][queue]") {
    auto dir = fs::temp_directory_path() / "md_dq_online";
    fs::remove_all(dir);
    FakeTransport tx;
    DurableQueue q(dir.string(), 8, &tx, 20);
    q.start();
    q.enqueue(ev("a")); q.enqueue(ev("b")); q.enqueue(ev("c"));
    for (int i = 0; i < 100 && tx.got.size() < 3; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    REQUIRE(tx.got == std::vector<std::string>{"a", "b", "c"});
    REQUIRE(q.pending() == 0);
    q.stop();
    fs::remove_all(dir);
}

TEST_CASE("DurableQueue caches offline then reflushes after restart", "[agent][queue]") {
    auto dir = fs::temp_directory_path() / "md_dq_offline";
    fs::remove_all(dir);
    {
        FakeTransport tx; tx.online = false;
        DurableQueue q(dir.string(), 8, &tx, 20);
        q.start();
        q.enqueue(ev("x")); q.enqueue(ev("y"));
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        REQUIRE(q.pending() == 2);
        REQUIRE(q.dropped() == 0);
        q.stop();
    }  // 重启
    {
        FakeTransport tx; tx.online = true;
        DurableQueue q(dir.string(), 8, &tx, 20);
        q.start();
        for (int i = 0; i < 100 && tx.got.size() < 2; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        REQUIRE(tx.got == std::vector<std::string>{"x", "y"});
        REQUIRE(q.pending() == 0);
        q.stop();
    }
    fs::remove_all(dir);
}

TEST_CASE("DurableQueue drops oldest over limit", "[agent][queue]") {
    auto dir = fs::temp_directory_path() / "md_dq_limit";
    fs::remove_all(dir);
    FakeTransport tx; tx.online = false;
    DurableQueue q(dir.string(), static_cast<size_t>(300), &tx, 20);   // 约 1~2 条
    q.start();
    for (int i = 0; i < 5; ++i) q.enqueue(ev("e" + std::to_string(i)));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    REQUIRE(q.dropped() > 0);
    REQUIRE(q.pending() <= 2);
    q.stop();
    fs::remove_all(dir);
}

TEST_CASE("DurableQueue discards vanished file without retry", "[agent][queue]") {
    auto dir = fs::temp_directory_path() / "md_dq_vanished";
    fs::remove_all(dir);
    FakeTransport tx; tx.online = true;
    DurableQueue q(dir.string(), 8, &tx, 20);
    q.enqueue(ev("gone"));
    REQUIRE(q.pending() == 1);
    // 模拟 enforce_limit 在 worker 取走前删除了文件
    for (const auto& entry : fs::directory_iterator(dir)) fs::remove(entry.path());
    q.start();
    for (int i = 0; i < 100 && q.pending() != 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    REQUIRE(q.pending() == 0);   // 条目被安全移除
    REQUIRE(tx.got.empty());     // 未发布、未当成传输失败无限重试
    q.stop();
    fs::remove_all(dir);
}

TEST_CASE("DurableQueue dedups identical event_id", "[agent][queue]") {
    auto dir = fs::temp_directory_path() / "md_dq_dedup";
    fs::remove_all(dir);
    FakeTransport tx; tx.online = false;
    DurableQueue q(dir.string(), 8, &tx, 20);
    q.start();
    q.enqueue(ev("same")); q.enqueue(ev("same"));
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    REQUIRE(q.pending() == 1);
    q.stop();
    fs::remove_all(dir);
}

TEST_CASE("DurableQueue attaches inline snapshot via provider", "[agent][queue]") {    auto dir = fs::temp_directory_path() / "md_dq_snap";
    fs::remove_all(dir);

    struct CapturingTransport : EventPublisher {
        std::atomic<bool> online{true};
        std::vector<DetectionEvent> events;
        std::mutex m;
        bool publish(const DetectionEvent& e) override {
            if (!online.load()) return false;
            std::lock_guard<std::mutex> lk(m);
            events.push_back(e);
            return true;
        }
        std::string name() const override { return "cap"; }
    };

    CapturingTransport tx;
    DurableQueue q(dir.string(), 8, &tx, 20);
    q.set_snapshot_provider([](int64_t, std::string* b64, int* w, int* h) {
        *b64 = "QUJD"; *w = 320; *h = 180; return true;
    });
    q.start();
    q.enqueue(ev("s1"));
    for (int i = 0; i < 100 && tx.events.empty(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    REQUIRE(tx.events.size() == 1);
    REQUIRE(tx.events[0].snapshot_data == "QUJD");
    REQUIRE(tx.events[0].snapshot_width == 320);
    q.stop();
    fs::remove_all(dir);
}

// 合同审计 HIGH：磁盘写失败（落盘失败 → 事件必然丢失）必须计数并可观测，不得静默丢弃。
// 用一个"父路径是普通文件"的非法队列目录构造必然失败的 write 路径。
TEST_CASE("DurableQueue counts disk write failures", "[agent][queue]") {
    auto dir = fs::temp_directory_path() / "md_dq_writefail";
    fs::remove_all(dir);
    fs::create_directories(dir);
    // dir/blocker 是普通文件：dir/blocker/q 作为队列目录无法创建，写 tmp 文件必然失败
    auto blocker = dir / "blocker";
    {
        std::ofstream f(blocker);
        f << "x";
    }
    FakeTransport tx;
    DurableQueue q((blocker / "q").string(), 8, &tx, 20);
    q.start();
    q.enqueue(ev("w1"));
    q.enqueue(ev("w2"));
    q.enqueue(ev("w3"));
    REQUIRE(q.write_failed() == 3);   // 旧实现静默 return，计数恒为 0
    REQUIRE(q.pending() == 0);
    REQUIRE(q.pending_bytes() == 0);
    q.stop();
    fs::remove_all(dir);
}

// 合同审计 HIGH：传输永久失败必须计数（send_failed）并可观测，不得静默累积。
TEST_CASE("DurableQueue counts transport failures", "[agent][queue]") {
    auto dir = fs::temp_directory_path() / "md_dq_sendfail";
    fs::remove_all(dir);
    FakeTransport tx;
    tx.online = false;
    DurableQueue q(dir.string(), 8, &tx, 20);
    q.start();
    q.enqueue(ev("s1"));
    for (int i = 0; i < 200 && q.send_failed() < 2; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    REQUIRE(q.send_failed() >= 2);   // 记录失败次数（旧实现无任何计数）
    REQUIRE(q.pending() == 1);       // 队首保留，按序重试
    REQUIRE(q.dropped() == 0);       // 传输失败 ≠ 超限丢弃，两者语义分明
    q.stop();
    fs::remove_all(dir);
}
