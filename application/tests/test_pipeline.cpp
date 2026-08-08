#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <chrono>
#include "pipeline.hpp"

#ifdef WITH_GPU
#include <cuda_runtime.h>
// P4 设备缓冲生命周期契约：decode_loop 把 CUVID 帧 D2D 拷贝到 PendingFrame 持有的
// 持久设备缓冲，raw 指针（y_plane_device/uv_plane_device）在 move 进队列/出队列后
// 必须仍指向移动后 shared_ptr 持有的同一设备缓冲（read_hw_frame_ 复用帧不可直接外传）。
TEST_CASE("PendingFrame GPU NV12 persistent buffer move propagation", "[pipeline][gpu]") {
    const size_t y_size = 640u * 360u;
    const size_t uv_size = y_size / 2;
    uint8_t* d = nullptr;
    if (cudaMalloc(&d, y_size + uv_size) != cudaSuccess) {
        WARN("CUDA unavailable; skipping device buffer test");
        return;
    }
    std::shared_ptr<uint8_t> buf(d, [](uint8_t* p) { cudaFree(p); });

    PendingFrame pf;
    pf.gpu_nv12 = buf;
    pf.y_plane_device = buf.get();
    pf.uv_plane_device = buf.get() + y_size;

    // 模拟 decode_loop push / process_loop pop 的 move 语义
    PendingFrame moved = std::move(pf);
    REQUIRE(moved.gpu_nv12.get() != nullptr);
    REQUIRE(moved.gpu_nv12 == buf);
    REQUIRE(moved.y_plane_device == moved.gpu_nv12.get());
    REQUIRE(moved.uv_plane_device == moved.gpu_nv12.get() + y_size);
    REQUIRE(static_cast<size_t>(moved.uv_plane_device - moved.y_plane_device) == y_size);
}

// D2D 拷贝把 strided CUVID 帧（行步长 > width，含对齐填充）打包进连续持久缓冲，
// 与 pipeline.cpp decode_loop 的 cudaMemcpy2D 逻辑一致。
TEST_CASE("PendingFrame D2D copy packs strided CUVID frame", "[pipeline][gpu]") {
    const int W = 64, H = 32;
    const int y_stride = W + 16;  // cuvid 对齐填充
    const size_t y_size = static_cast<size_t>(W) * H;

    uint8_t* cuvid_y = nullptr;
    uint8_t* cuvid_uv = nullptr;
    if (cudaMalloc(&cuvid_y, static_cast<size_t>(y_stride) * H) != cudaSuccess ||
        cudaMalloc(&cuvid_uv, static_cast<size_t>(y_stride) * H / 2) != cudaSuccess) {
        WARN("CUDA unavailable; skipping D2D copy test");
        if (cuvid_y) cudaFree(cuvid_y);
        return;
    }

    // 构造 strided 表面：行内连续 W 字节，行尾填充
    std::vector<uint8_t> host_pat(y_size);
    std::vector<uint8_t> strided_host(static_cast<size_t>(y_stride) * H);
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c)
            host_pat[static_cast<size_t>(r) * W + c] = static_cast<uint8_t>((r * W + c) & 0xFF);
        std::memcpy(&strided_host[static_cast<size_t>(r) * y_stride],
                    &host_pat[static_cast<size_t>(r) * W], W);
    }
    REQUIRE(cudaMemcpy(cuvid_y, strided_host.data(), strided_host.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // 模拟 decode_loop 的 D2D 拷贝：dpitch=width，spitch=cuvid stride
    uint8_t* raw = nullptr;
    REQUIRE(cudaMalloc(&raw, y_size + y_size / 2) == cudaSuccess);
    std::shared_ptr<uint8_t> dbuf(raw, [](uint8_t* p) { cudaFree(p); });
    const cudaError_t ey = cudaMemcpy2D(dbuf.get(), W, cuvid_y, y_stride,
                                        W, H, cudaMemcpyDeviceToDevice);
    REQUIRE(ey == cudaSuccess);

    std::vector<uint8_t> got(y_size);
    REQUIRE(cudaMemcpy(got.data(), dbuf.get(), y_size, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(std::memcmp(got.data(), host_pat.data(), y_size) == 0);

    cudaFree(cuvid_y);
    cudaFree(cuvid_uv);
}
#endif

TEST_CASE("Pipeline construct/destroy", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "test";
    cfg.input_url = "rtsp://dummy";
    cfg.output_url = "rtsp://dummy-out";
    Pipeline pipe(cfg);
    REQUIRE_FALSE(pipe.is_running());
}

TEST_CASE("Pipeline task_id", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "cam01";
    Pipeline pipe(cfg);
    REQUIRE(pipe.task_id() == "cam01");
}

TEST_CASE("Pipeline start is async", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "async_test";
    cfg.input_url = "rtsp://nonexistent";
    cfg.output_url = "rtsp://dummy-out";
    Pipeline pipe(cfg);
    // start() 返回 true，初始化在后台进行
    bool ok = pipe.start();
    REQUIRE(ok);
    REQUIRE(pipe.is_running());
    // 初始化最终会失败（无 RTSP 流），线程会退出
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    pipe.stop();
}

TEST_CASE("Pipeline double start", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "double";
    Pipeline pipe(cfg);
    REQUIRE(pipe.start());
    REQUIRE(pipe.start());  // second start is no-op
    pipe.stop();
}

TEST_CASE("Pipeline concurrent start only creates one decode thread", "[pipeline][stress]") {
    TaskConfig cfg;
    cfg.id = "concurrent_start";
    cfg.input_url = "rtsp://127.0.0.1:1/live/x";  // 不存在的端口，快速失败
    cfg.output_url = "rtsp://out";
    cfg.decoder.timeout_us = 500000;
    for (int round = 0; round < 10; ++round) {
        Pipeline pipe(cfg);
        std::thread t1([&] { pipe.start(); });
        std::thread t2([&] { pipe.start(); });
        std::thread t3([&] { pipe.start(); });
        t1.join(); t2.join(); t3.join();
        pipe.stop();  // 若重复创建 decode 线程会泄漏/崩溃
    }
    REQUIRE(true);
}

TEST_CASE("Pipeline stop without start", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "safe_stop";
    Pipeline pipe(cfg);
    REQUIRE_NOTHROW(pipe.stop());
}

TEST_CASE("Pipeline start then stop quickly", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "fast_stop";
    cfg.input_url = "rtsp://dummy";
    cfg.output_url = "rtsp://out";
    Pipeline pipe(cfg);
    REQUIRE(pipe.start());
    pipe.stop();
    REQUIRE_FALSE(pipe.is_running());
}

TEST_CASE("Pipeline model operations without start", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "model_ops";
    Pipeline pipe(cfg);

    ModelConfig mcfg;
    mcfg.name = "det";
    mcfg.path = "/nonexistent.onnx";
    REQUIRE_FALSE(pipe.add_model(mcfg));
    REQUIRE_FALSE(pipe.remove_model("det"));
    REQUIRE_FALSE(pipe.update_model("det", mcfg));
}

TEST_CASE("Pipeline config", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "config_test";
    cfg.input_url = "rtsp://cam";
    cfg.output_url = "rtmp://push";
    Pipeline pipe(cfg);
    const auto& c = pipe.config();
    REQUIRE(c.id == "config_test");
    REQUIRE(c.input_url == "rtsp://cam");
    REQUIRE(c.output_url == "rtmp://push");
}

TEST_CASE("Pipeline real model and stream", "[pipeline][integration]") {
    TaskConfig cfg;
    cfg.id = "integration_test";
    cfg.input_url = "rtsp://127.0.0.1:554/live/obs_test3";
    cfg.output_url = "E:/CLionProjects/ModelDeploy/build/bin/integration_out.mp4";
    cfg.draw.show_label = true;
    cfg.draw.show_score = true;

    ModelConfig mcfg;
    mcfg.name = "yolo11n";
    mcfg.type = "detection";
    mcfg.path = "E:/CLionProjects/ModelDeploy/test_data/test_models/yolo11n_nms.onnx";
    mcfg.backend = "ort";
    mcfg.device = "gpu";
    mcfg.confidence_threshold = 0.5f;
    mcfg.input_size = {640, 640};
    mcfg.interval = 1;
    cfg.models.push_back(mcfg);

    Pipeline pipe(cfg);
    // start() is async, returns immediately
    REQUIRE_NOTHROW(pipe.start());
    // Wait for initialization to complete or fail
    std::this_thread::sleep_for(std::chrono::seconds(3));
    pipe.stop();
}
