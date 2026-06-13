#include "pipeline.hpp"
#include <iostream>

using namespace modeldeploy::vision;

Pipeline::Pipeline(TaskConfig cfg)
    : cfg_(std::move(cfg)) {
}

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::start() {
    if (running_) return true;

    // 异步初始化：在 pipeline 线程中做重量级工作（连接 RTSP、加载模型）
    // start() 立即返回，不阻塞 HTTP handler
    initialized_ = false;
    running_ = true;
    pipeline_thread_ = std::thread(&Pipeline::pipeline_loop, this);
    return true;
}

void Pipeline::stop() {
    running_ = false;
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

void Pipeline::pipeline_loop() {
    try {
        // === 初始化阶段 ===

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

        // 3) StreamEncoder (deferred open)
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
        std::cout << "[Pipeline] Started: " << cfg_.id
                  << " (" << cfg_.input_url << " \u2192 " << cfg_.output_url << ")" << std::endl;
        stats_.start();

        // === 保持线程存活（解码器在自己的线程中运行） ===
        while (running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
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

bool Pipeline::on_decoded_frame(const DecodedFrame& frame) {
    if (!running_) return false;

    try {
        if (!encoder_opened_) {
            if (encoder_->open(cfg_.output_url, frame.width, frame.height)) {
                encoder_->start_async();
                encoder_opened_ = true;
            } else {
                std::cerr << "[Pipeline] Encoder open failed" << std::endl;
                return false;
            }
        }

        auto t0 = std::chrono::steady_clock::now();

        std::vector<InferResult> results;
        ImageData frame_bgr;
        infer_group_->run_models(frame.y_plane, frame.uv_plane,
                                  frame.width, frame.height,
                                  frame.y_step, frame.uv_step,
                                  &results, &frame_bgr);

        auto t1 = std::chrono::steady_clock::now();

        draw_engine_->draw(frame_bgr, results);

        auto t2 = std::chrono::steady_clock::now();

        encoder_->encode_async(frame_bgr);

        auto t3 = std::chrono::steady_clock::now();

        stats_.record_frame(0,
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(),
            std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count());
    } catch (const std::exception& e) {
        std::cerr << "[Pipeline] Frame error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[Pipeline] Frame unknown error" << std::endl;
    }

    return running_.load();
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
