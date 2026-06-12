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

    // 1) InferGroup
    infer_group_ = std::make_unique<InferGroup>(cfg_);
    if (!infer_group_->init()) {
        std::cerr << "[Pipeline] InferGroup init failed" << std::endl;
        return false;
    }

    // 2) DrawEngine
    draw_engine_ = std::make_unique<DrawEngine>(cfg_.draw);

    // 3) StreamEncoder (deferred open until first frame provides resolution)
    encoder_ = std::make_unique<StreamEncoder>(25);

    // 4) StreamDecoder
    decoder_ = std::make_unique<StreamDecoder>(cfg_.decoder);
    decoder_->set_callback([this](const DecodedFrame& frame) -> bool {
        return on_decoded_frame(frame);
    });
    if (!decoder_->open(cfg_.input_url)) {
        std::cerr << "[Pipeline] Decoder open failed" << std::endl;
        return false;
    }

    running_ = true;
    pipeline_thread_ = std::thread(&Pipeline::pipeline_loop, this);

    if (!decoder_->start()) {
        std::cerr << "[Pipeline] Decoder start failed" << std::endl;
        stop();
        return false;
    }

    std::cout << "[Pipeline] Started: " << cfg_.id
              << " (" << cfg_.input_url << " \u2192 " << cfg_.output_url << ")" << std::endl;
    stats_.start();
    return true;
}

void Pipeline::stop() {
    running_ = false;
    if (decoder_) decoder_->stop();
    if (pipeline_thread_.joinable())
        pipeline_thread_.join();
    if (encoder_) encoder_->stop_async();
    if (encoder_) encoder_->close();

    stats_.print();

    decoder_.reset();
    infer_group_.reset();
    draw_engine_.reset();
    encoder_.reset();
    encoder_opened_ = false;
}

bool Pipeline::on_decoded_frame(const DecodedFrame& frame) {
    if (!running_) return false;

    // 首次解码到帧时打开编码器（获取流的分辨率）
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

    // 1) 推理
    std::vector<InferResult> results;
    ImageData frame_bgr;
    infer_group_->run_models(frame.y_plane, frame.uv_plane,
                              frame.width, frame.height,
                              frame.y_step, frame.uv_step,
                              &results, &frame_bgr);

    auto t1 = std::chrono::steady_clock::now();

    // 2) 绘制
    draw_engine_->draw(frame_bgr, results);

    auto t2 = std::chrono::steady_clock::now();

    // 3) 编码推送
    encoder_->encode_async(frame_bgr);

    auto t3 = std::chrono::steady_clock::now();

    auto inf_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto drw_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    auto enc_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    stats_.record_frame(0, inf_us, drw_us, enc_us);

    return running_.load();
}

void Pipeline::pipeline_loop() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
