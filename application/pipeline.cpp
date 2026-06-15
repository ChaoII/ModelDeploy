#include "pipeline.hpp"
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <windows.h>
#include <eh.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace modeldeploy::vision;

static bool is_network_url(const std::string& url) {
    return url.find("rtsp://") == 0 || url.find("rtmp://") == 0 ||
           url.find("http://") == 0 || url.find("https://") == 0 ||
           url.find("udp://") == 0 || url.find("tcp://") == 0;
}

Pipeline::Pipeline(TaskConfig cfg)
    : cfg_(std::move(cfg)),
      block_on_queue_full_(!is_network_url(cfg_.input_url)) {
}

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::start() {
    if (running_.load()) return true;
    initialized_ = false;
    init_error_.clear();
    running_ = true;
    pipeline_thread_ = std::thread(&Pipeline::pipeline_loop, this);
    return true;
}

void Pipeline::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        // 已经停止或从未启动
        if (pipeline_thread_.joinable())
            pipeline_thread_.join();
        release_resources();
        return;
    }

    // 唤醒流水线线程与可能在等待队列的解码回调
    queue_cv_.notify_all();
    queue_not_full_cv_.notify_all();

    // 等待流水线线程结束（它拥有 decoder/encoder 的生命周期）
    if (pipeline_thread_.joinable())
        pipeline_thread_.join();

    // 线程结束后，再释放剩余资源
    release_resources();
    stats_.print();
}

void Pipeline::release_resources() {
    // 线程结束后才能安全释放
    decoder_.reset();
    infer_group_.reset();
    draw_engine_.reset();
    if (encoder_) {
        encoder_->stop_async();
        encoder_->close();
        encoder_.reset();
    }
    encoder_opened_ = false;
    initialized_ = false;

    // 清空帧队列
    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        std::queue<PendingFrame> empty;
        frame_queue_.swap(empty);
    }
}

// ── 解码回调（解码线程） ─────────────────────

bool Pipeline::on_decoded_frame(const DecodedFrame& frame) {
    if (!running_.load()) return false;

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
            std::unique_lock<std::mutex> lock(queue_mtx_);
            if (block_on_queue_full_) {
                // 文件输入：阻塞等待队列有空间，避免丢帧
                queue_not_full_cv_.wait_for(lock, std::chrono::milliseconds(200),
                    [this]() { return frame_queue_.size() < max_queue_size_ || !running_.load(); });
                if (!running_.load()) return false;
            } else if (frame_queue_.size() >= max_queue_size_) {
                return running_.load(); // 实时流：跳帧保持低延迟
            }
            if (frame_queue_.size() >= max_queue_size_) {
                return running_.load(); // 被唤醒后仍满（停止时）
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

static void seh_translater(unsigned int, struct _EXCEPTION_POINTERS*) {
    throw std::runtime_error("SEH exception");
}

void Pipeline::pipeline_loop() {
    _set_se_translator(seh_translater);

    // 确保本线程的 CUDA 上下文与主线程一致
    cudaSetDevice(0);

    try {
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
        encoder_ = std::make_unique<StreamEncoder>(cfg_.encoder);

        // 4) StreamDecoder
        decoder_ = std::make_unique<StreamDecoder>(cfg_.decoder);
        decoder_->set_callback([this](const DecodedFrame& frame) -> bool {
            return on_decoded_frame(frame);
        });

        if (!decoder_->open(cfg_.input_url)) {
            init_error_ = "Decoder open failed: " + cfg_.input_url;
            std::cerr << "[Pipeline] " << init_error_ << std::endl;
            running_ = false;
            return;
        }

        if (!decoder_->start()) {
            init_error_ = "Decoder start failed";
            std::cerr << "[Pipeline] " << init_error_ << std::endl;
            running_ = false;
            return;
        }

        initialized_ = true;
        std::cout << "[Pipeline] Running: " << cfg_.id << std::endl;
        stats_.start();

        // 主循环：处理队列中的帧
        while (running_.load()) {
            PendingFrame pf;
            {
                std::unique_lock<std::mutex> lock(queue_mtx_);
                queue_cv_.wait_for(lock, std::chrono::milliseconds(500),
                    [this]() { return !frame_queue_.empty() || !running_.load(); });
                if (!running_.load()) break;
                if (frame_queue_.empty()) continue;
                pf = std::move(frame_queue_.front());
                frame_queue_.pop();
            }
            queue_not_full_cv_.notify_one();

            process_frame(pf.nv12_data.data(), pf.width, pf.height);
        }

    } catch (const std::exception& e) {
        init_error_ = e.what();
        std::cerr << "[Pipeline] Fatal: " << init_error_ << std::endl;
    } catch (...) {
        init_error_ = "unknown error";
        std::cerr << "[Pipeline] Fatal: unknown error" << std::endl;
    }

    // 流水线线程自行停止解码与编码
    if (decoder_) decoder_->stop();
    if (encoder_) {
        encoder_->stop_async();
        encoder_->close();
    }

    running_ = false;
}

// ── 单帧处理（流水线线程，有 CUDA 上下文） ─────

bool Pipeline::process_frame(const uint8_t* nv12, int width, int height) {
    try {
        if (!encoder_opened_.load()) {
            if (encoder_->open(cfg_.output_url, width, height)) {
                encoder_->start_async();
                encoder_opened_ = true;
            } else {
                std::cerr << "[Pipeline] Encoder open failed" << std::endl;
                return false;
            }
        }

        auto t0 = std::chrono::steady_clock::now();

        // 推理组内部完成 NV12→BGR，并输出可直接绘制/编码的 BGR 图像
        std::vector<InferResult> results;
        ImageData bgr_image;
        if (!infer_group_->run_models(const_cast<uint8_t*>(nv12),
                                       const_cast<uint8_t*>(nv12 + width * height),
                                       width, height, width, width,
                                       &results, &bgr_image)) {
            // 即使推理失败也继续编码原始帧，避免画面卡死
        }

        auto t1 = std::chrono::steady_clock::now();

        // 绘制
        if (!bgr_image.empty()) {
            draw_engine_->draw(bgr_image, results);
        }

        auto t2 = std::chrono::steady_clock::now();

        // 编码
        if (!bgr_image.empty()) {
            encoder_->encode_async(bgr_image);

            // 缓存最新一帧供快照接口使用（仅保存 BGR 原始数据）
            try {
                cv::Mat mat;
                bgr_image.to_mat(mat, true);
                if (!mat.empty()) {
                    cv::Mat bgr;
                    if (mat.channels() == 4)      cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
                    else if (mat.channels() == 1) cv::cvtColor(mat, bgr, cv::COLOR_GRAY2BGR);
                    else                           bgr = mat;
                    std::lock_guard<std::mutex> lock(snapshot_mtx_);
                    latest_w_ = bgr.cols;
                    latest_h_ = bgr.rows;
                    size_t sz = static_cast<size_t>(bgr.cols) * bgr.rows * 3;
                    latest_bgr_data_.assign(bgr.data, bgr.data + sz);
                }
            } catch (...) {}
        }

        auto t3 = std::chrono::steady_clock::now();

        stats_.record_frame(0,
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(),
            std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count());

        return true;
    } catch (...) {
        std::cerr << "[Pipeline] process_frame error" << std::endl;
        return false;
    }
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

bool Pipeline::latest_jpeg(std::vector<uint8_t>* out, int quality) {
    if (!out) return false;
    cv::Mat bgr;
    {
        std::lock_guard<std::mutex> lock(snapshot_mtx_);
        if (latest_bgr_data_.empty() || latest_w_ <= 0 || latest_h_ <= 0) return false;
        bgr = cv::Mat(latest_h_, latest_w_, CV_8UC3, latest_bgr_data_.data()).clone();
    }
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, quality };
    return cv::imencode(".jpg", bgr, *out, params);
}
