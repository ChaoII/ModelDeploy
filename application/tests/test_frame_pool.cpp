#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include "frame_pool.hpp"

static bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

TEST_CASE("FramePool basic acquire/release", "[frame_pool]") {
    if (!cuda_available()) SKIP("CUDA not available");
    FramePool pool(4);
    auto* buf = pool.acquire(1024);
    REQUIRE(buf != nullptr);
    REQUIRE(pool.cached_count() == 0);

    pool.release(buf);
    REQUIRE(pool.cached_count() == 1);
    pool.clear();
    REQUIRE(pool.cached_count() == 0);
}

TEST_CASE("FramePool reuse same size", "[frame_pool]") {
    if (!cuda_available()) SKIP("CUDA not available");
    FramePool pool(4);
    auto* b1 = pool.acquire(4096);
    pool.release(b1);

    auto* b2 = pool.acquire(4096);
    REQUIRE(b2 == b1);
    pool.release(b2);
}

TEST_CASE("FramePool reuse larger size", "[frame_pool]") {
    if (!cuda_available()) SKIP("CUDA not available");
    FramePool pool(4);
    auto* b1 = pool.acquire(8192);
    pool.release(b1);

    auto* b2 = pool.acquire(4096);
    REQUIRE(b2 == b1);
    pool.release(b2);
}

TEST_CASE("FramePool allocate larger when cached too small", "[frame_pool]") {
    if (!cuda_available()) SKIP("CUDA not available");
    FramePool pool(4);
    auto* b1 = pool.acquire(1024);
    pool.release(b1);

    auto* b2 = pool.acquire(8192);
    REQUIRE(b2 != nullptr);
    REQUIRE(b2 != b1);
    pool.release(b2);
    pool.release(b1);
}

TEST_CASE("FramePool pool size limit", "[frame_pool]") {
    if (!cuda_available()) SKIP("CUDA not available");
    FramePool pool(2);
    auto* b1 = pool.acquire(100);
    auto* b2 = pool.acquire(100);
    auto* b3 = pool.acquire(100);

    pool.release(b1);
    pool.release(b2);
    pool.release(b3);

    REQUIRE(pool.cached_count() == 2);
    pool.clear();
}

TEST_CASE("FramePool null release does nothing", "[frame_pool]") {
    FramePool pool(4);
    REQUIRE_NOTHROW(pool.release(nullptr));
}

TEST_CASE("FramePool clear with active allocations", "[frame_pool]") {
    if (!cuda_available()) SKIP("CUDA not available");
    FramePool pool(4);
    auto* b1 = pool.acquire(100);
    auto* b2 = pool.acquire(200);
    REQUIRE(pool.cached_count() == 0);

    pool.release(b1);
    REQUIRE(pool.cached_count() == 1);

    pool.clear();
    REQUIRE(pool.cached_count() == 0);
}

TEST_CASE("FramePool multithreaded", "[frame_pool]") {
    if (!cuda_available()) SKIP("CUDA not available");
    FramePool pool(8);
    std::vector<uint8_t> host_buf(1024, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 50; ++i) {
                auto* dev = pool.acquire(1024);
                if (!dev) continue;
                cudaMemcpy(dev, host_buf.data(), 1024, cudaMemcpyHostToDevice);
                cudaMemcpy(host_buf.data(), dev, 1024, cudaMemcpyDeviceToHost);
                pool.release(dev);
            }
        });
    }
    for (auto& th : threads) th.join();
    REQUIRE(pool.cached_count() <= 8);
    pool.clear();
}
