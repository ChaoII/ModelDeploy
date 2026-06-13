#include "pipeline.hpp"
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>

using namespace modeldeploy::vision;

Pipeline::Pipeline(TaskConfig cfg)
    : cfg_(std::move(cfg)) {
}

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::start() {
    if (running_) return true;
    initialized_ = false;
    running_ = true;
    pipeline_thread_ = std::thread(&Pipeline::pipeline_loop, this);
    return true;
}

void Pipeline::stop() {
    running_ = false;
    queue_cv_.notify_all();
    if (decoder_) {
        decoder_->stop();
        decoder_.reset();
    }
    if (pipeline_thread_.joinable())
        pipeline_thread_.join();
    if (encoder_) {
        encoder_->stop_async();
        encoder_->close();
        encoder_.reset();
    }
    infer_group_.reset();
    draw_engine_.reset();
    encoder_opened_ = false;
    initialized_ = false;
    stats_.print();
}

// ── 解码回调（解码线程） ─────────────────────

bool Pipeline::on_decoded_frame(const DecodedFrame& frame) {
    if (!running_) return false;

    try {
        // 逐行拷贝 NV12 帧数据到连续 buffer
        PendingFrame pf;
        pf.width = frame.width;
        pf.height = frame.height;
        size_t y_size = static_cast<size_t>(frame.height) * frame.width;
        size_t uv_size = y_size / 2;
        pf.nv12_data.resize(y_size + uv_size);

        uint8_t* dst = pf.nv12_data.data();
        for (int row = 0; row < frame.height; ++row) {
            memcpy(dst + row * frame.width,
                   frame.y_plane + row * frame.y_step,
                   frame.width);
        }
        int uv_h = frame.height / 2;
        for (int row = 0; row < uv_h; ++row) {
            memcpy(dst + y_size + row * frame.width,
                   frame.uv_plane + row * frame.uv_step,
                   frame.width);
        }

        // 推入队列，唤醒流水线线程
        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            if (frame_queue_.size() >= max_queue_size_) {
                return running_.load(); // skip frame
            }
            frame_queue_.push(std::move(pf));
        }
        queue_cv_.notify_one();
    } catch (const std::exception& e) {
        std::cerr << "[Pipeline] Callback error: " << e.what() << std::endl;
    }

    return running_.load();
}

// ── 流水线主循环（流水线线程） ─────────────────

void Pipeline::pipeline_loop() {
    try {
        cudaSetDevice(0);

        // 1) InferGroup
        infer_group_ = std::make_unique<InferGroup>(cfg_);
        if (!infer_group_->init()) {
            init_error_ = "InferGroup init failed";
            std::cerr << "[Pipeline] " << init_error_ << std::endl;
            running_ = false;
            return;
        }

        // 2) DrawEngine
        draw_engine_ = std::make_unique<DrawEngine>(cfg_.draw);

        // 3) StreamEncoder
        encoder_ = std::make_unique<StreamEncoder>(25);

        // 4) StreamDecoder
        decoder_ = std::make_unique<StreamDecoder>(cfg_.decoder);
        decoder_->set_callback([this](const DecodedFrame& frame) -> bool {
            return on_decoded_frame(frame);
        });

        if (!decoder_->open(cfg_.input_url)) {
            init_error_ = "Decoder open failed";
            std::cerr << "[Pipeline] " << init_error_ << std::endl;
            running_ = false;
            return;
        }

        if (!decoder_->start()) {
            init_error_ = "Decoder start failed";
            std::cerr << "[Pipeline] " << init_error_ << std::endl;
            decoder_.reset();
            infer_group_.reset();
            draw_engine_.reset();
            encoder_.reset();
            running_ = false;
            return;
        }

        initialized_ = true;
        std::cout << "[Pipeline] Running: " << cfg_.id << std::endl;
        stats_.start();

        // 主循环：处理队列中的帧
        while (running_) {
            PendingFrame pf;
            {
                std::unique_lock<std::mutex> lock(queue_mtx_);
                queue_cv_.wait_for(lock, std::chrono::milliseconds(500),
                    [this]() { return !frame_queue_.empty() || !running_; });
                if (!running_) break;
                if (frame_queue_.empty()) continue;
                pf = std::move(frame_queue_.front());
                frame_queue_.pop();
            }

            process_frame(pf.nv12_data.data(), pf.width, pf.height);
        }

    } catch (const std::exception& e) {
        init_error_ = e.what();
        std::cerr << "[Pipeline] Fatal: " << init_error_ << std::endl;
    } catch (...) {
        init_error_ = "unknown error";
        std::cerr << "[Pipeline] Fatal: unknown error" << std::endl;
    }

    running_ = false;
    if (decoder_) { decoder_->stop(); decoder_.reset(); }
    infer_group_.reset();
    draw_engine_.reset();
    if (encoder_) { encoder_->stop_async(); encoder_->close(); encoder_.reset(); }
}

// ── 单帧处理（流水线线程，有 CUDA 上下文） ─────

bool Pipeline::process_frame(const uint8_t* nv12, int width, int height) {
    if (!encoder_opened_) {
        if (encoder_->open(cfg_.output_url, width, height)) {
            encoder_->start_async();
            encoder_opened_ = true;
        } else {
            std::cerr << "[Pipeline] Encoder open failed" << std::endl;
            return false;
        }
    }

    auto t0 = std::chrono::steady_clock::now();

    auto nv12_image = ImageData::from_raw(const_cast<uint8_t*>(nv12), width, height,
                                           MdImageType::NV12, true);
    auto bgr_image = ImageData::cvt_color(nv12_image, ColorConvertType::CVT_NV122PA_BGR);

    // 推理
    std::vector<InferResult> results;
    infer_group_->run_models(const_cast<uint8_t*>(nv12),
                              const_cast<uint8_t*>(nv12 + width * height),
                              width, height, width, width,
                              &results, nullptr);

    auto t1 = std::chrono::steady_clock::now();

    // 绘制
    draw_engine_->draw(bgr_image, results);

    auto t2 = std::chrono::steady_clock::now();

    // 编码
    encoder_->encode_async(bgr_image);

    auto t3 = std::chrono::steady_clock::now();

    stats_.record_frame(0,
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(),
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count());

    return true;
}

bool Pipeline::add_model(const ModelConfig& mcfg) {
    if (!infer_group_) return false;
    return infer_group_->add_model(mcfg);
}

bool Pipeline::remove_model(const std::string& name) {
    if (!infer_group_) return false;
    return infer_group_->remove_model(name);
}

bool Pipeline::update_model(const std::string& name, const ModelConfig& mcfg) {
    if (!infer_group_) return false;
    return infer_group_->update_model(name, mcfg);
}
