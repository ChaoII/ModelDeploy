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

Pipeline::Pipeline(TaskConfig cfg, StreamHub* hub, ModelFactory model_factory, BatchScheduler* batch_scheduler)
    : cfg_(std::move(cfg)),
      hub_(hub),
      model_factory_(std::move(model_factory)),
      batch_scheduler_(batch_scheduler),
      block_on_in_full_(!is_network_url(cfg_.input_url)) {
    in_max_size_ = calc_in_queue_size();
    out_max_size_ = 3;
}

size_t Pipeline::calc_in_queue_size() const {
    // 网络流：小队列保持低延迟，丢老帧
    // 文件源：大队列吃满吞吐
    if (is_network_url(cfg_.input_url)) {
        return 2; // 只缓 2 帧，足以掩盖偶发抖动
    }
    return 8;
}

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::start() {
    if (running_.load()) return true;
    stopped_ = false;
    initialized_ = false;
    init_error_.clear();
    running_ = true;

    // 三段流水线主线程
    decode_thread_ = std::thread([this]() {
        try {
            decode_loop();
        } catch (const std::exception& e) {
            init_error_ = e.what();
            std::cerr << "[Pipeline-decode] Fatal: " << e.what() << std::endl;
        } catch (...) {
            init_error_ = "decode unknown error";
            std::cerr << "[Pipeline-decode] Fatal: unknown" << std::endl;
        }
        running_ = false;
        in_cv_.notify_all();
        out_cv_.notify_all();
    });
    return true;
}

void Pipeline::stop() {
    bool expected = true;
    if (!stopped_.load()) {
        stopped_ = true;
    } else {
        return;
    }
    if (!running_.compare_exchange_strong(expected, false)) {
        running_ = false;
    }
    in_cv_.notify_all();
    out_cv_.notify_all();
    if (decode_thread_.joinable())  decode_thread_.join();
    if (process_thread_.joinable()) process_thread_.join();
    if (encode_thread_.joinable())  encode_thread_.join();
    release_resources();
    stats_.print();
}

void Pipeline::release_resources() {
    if (shared_source_ && shared_token_) {
        shared_source_->unsubscribe(shared_token_);
        shared_token_ = 0;
    }
    shared_source_.reset();
    decoder_.reset();
    infer_group_.reset();
    draw_engine_.reset();
    if (encoder_) {
        encoder_->close();
        encoder_.reset();
    }
    encoder_opened_ = false;
    initialized_ = false;
    cached_results_.clear();

    {
        std::lock_guard<std::mutex> lock(in_mtx_);
        std::queue<PendingFrame> empty;
        in_queue_.swap(empty);
    }
    {
        std::lock_guard<std::mutex> lock(out_mtx_);
        std::queue<EncodedFrame> empty;
        out_queue_.swap(empty);
    }
}

// ── 解码回调（StreamHub 模式：跨 pipeline 共享解码器） ──

bool Pipeline::on_shared_frame(const std::shared_ptr<BroadcastFrame>& frame) {
    if (!running_.load() || !frame) return false;
    PendingFrame pf;
    pf.width = frame->width;
    pf.height = frame->height;
    pf.pts = frame->pts;
    pf.wall_time_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    pf.shared_frame = frame;
    push_in_queue(std::move(pf));
    return running_.load();
}

void Pipeline::push_in_queue(PendingFrame&& pf) {
    std::unique_lock<std::mutex> lock(in_mtx_);
    if (in_queue_.size() >= in_max_size_) {
        if (block_on_in_full_) {
            // 文件源：等待消费，避免丢帧
            in_cv_.wait_for(lock, std::chrono::milliseconds(200),
                [this]() { return in_queue_.size() < in_max_size_ || !running_.load(); });
            if (!running_.load() || in_queue_.size() >= in_max_size_) return;
        } else {
            // 网络流：丢最旧的，保最新（保头丢尾，确保延迟可控）
            in_queue_.pop();
        }
    }
    in_queue_.push(std::move(pf));
    lock.unlock();
    in_cv_.notify_one();
}

// ── 解码段（解码线程） ──

static void seh_translater(unsigned int, struct _EXCEPTION_POINTERS*) {
    throw std::runtime_error("SEH exception");
}

void Pipeline::decode_loop() {
    _set_se_translator(seh_translater);
    cudaSetDevice(0);

    // ── 1) 初始化推理组 + 绘制 ──
    infer_group_ = std::make_unique<InferGroup>(cfg_, model_factory_);
    if (!infer_group_->init()) {
        init_error_ = "InferGroup init failed";
        std::cerr << "[Pipeline] " << init_error_ << std::endl;
        running_ = false;
        return;
    }
    draw_engine_ = std::make_unique<DrawEngine>(cfg_.draw);
    if (cfg_.enable_preview) {
        encoder_ = std::make_unique<StreamEncoder>(cfg_.encoder);
    }

    // ── 2) 启动推理线程 ──
    process_thread_ = std::thread([this]() {
        cudaSetDevice(0);
        _set_se_translator(seh_translater);
        try { process_loop(); }
        catch (const std::exception& e) { std::cerr << "[Pipeline-process] " << e.what() << std::endl; }
        catch (...) { std::cerr << "[Pipeline-process] unknown" << std::endl; }
        out_cv_.notify_all();
    });
    if (cfg_.enable_preview) {
        encode_thread_ = std::thread([this]() {
            _set_se_translator(seh_translater);
            try { encode_loop(); }
            catch (const std::exception& e) { std::cerr << "[Pipeline-encode] " << e.what() << std::endl; }
            catch (...) { std::cerr << "[Pipeline-encode] unknown" << std::endl; }
        });
    }

    // ── 3) 解码：StreamHub 共享 or 独占 ──
    if (hub_ && is_network_url(cfg_.input_url)) {
        shared_source_ = hub_->acquire(cfg_.input_url, cfg_.decoder);
        if (!shared_source_) {
            init_error_ = "StreamHub acquire failed";
            running_ = false;
            return;
        }
        shared_token_ = shared_source_->subscribe(
            [this](const std::shared_ptr<BroadcastFrame>& f) { this->on_shared_frame(f); }
        );
        if (!shared_source_->is_initialized()) {
            init_error_ = shared_source_->init_error();
            std::cerr << "[Pipeline] " << init_error_ << std::endl;
            running_ = false;
            return;
        }
        initialized_ = true;
        source_fps_ = shared_source_->fps();
        std::cout << "[Pipeline] " << cfg_.id << " -> shared source fps=" << source_fps_ << std::endl;
        initialized_ = true;
        std::cout << "[Pipeline] Running (shared): " << cfg_.id << std::endl;
        stats_.start();
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } else {
        decoder_ = std::make_unique<StreamDecoder>(cfg_.decoder);
        if (!decoder_->open(cfg_.input_url)) {
            init_error_ = "Decoder open failed: " + cfg_.input_url;
            std::cerr << "[Pipeline] " << init_error_ << std::endl;
            running_ = false;
            return;
        }
        source_fps_ = decoder_->fps();
        std::cout << "[Pipeline] Source FPS: " << source_fps_ << std::endl;
        initialized_ = true;
        std::cout << "[Pipeline] Running (own): " << cfg_.id << std::endl;
        stats_.start();

        DecodedFrame raw;
        while (running_.load() && decoder_->read_one_frame(&raw)) {
            auto dec_t0 = std::chrono::steady_clock::now();
            PendingFrame pf;
            pf.width = raw.width;
            pf.height = raw.height;
            pf.pts = raw.pts;
            pf.wall_time_sec = std::chrono::duration<double>(
                dec_t0.time_since_epoch()).count();
            const size_t y_size = static_cast<size_t>(raw.height) * raw.width;
            const size_t uv_size = y_size / 2;
            pf.nv12_data.resize(y_size + uv_size);
            uint8_t* dst = pf.nv12_data.data();
            if (raw.y_step == raw.width && raw.uv_step == raw.width) {
                std::memcpy(dst, raw.y_plane, y_size + uv_size);
            } else {
                for (int row = 0; row < raw.height; ++row)
                    std::memcpy(dst + row * raw.width,
                                raw.y_plane + row * raw.y_step, raw.width);
                const int uv_h = raw.height / 2;
                for (int row = 0; row < uv_h; ++row)
                    std::memcpy(dst + y_size + row * raw.width,
                                raw.uv_plane + row * raw.uv_step, raw.width);
            }
            auto dec_t1 = std::chrono::steady_clock::now();
            last_decode_us_ = std::chrono::duration_cast<std::chrono::microseconds>(dec_t1 - dec_t0).count();
            push_in_queue(std::move(pf));
        }
    }

    // 通知后续段退出
    running_ = false;
    in_cv_.notify_all();
    out_cv_.notify_all();
    if (decoder_) decoder_->stop();
}

// ── 推理+绘制段（推理线程） ──

void Pipeline::process_loop() {
    int64_t snapshot_counter = 0;
    bool has_cached_results = false;
    auto t_last = std::chrono::steady_clock::now();
    while (running_.load()) {
        PendingFrame pf;
        {
            std::unique_lock<std::mutex> lock(in_mtx_);
            in_cv_.wait(lock, [this]() {
                return !in_queue_.empty() || !running_.load();
            });
            if (!running_.load() && in_queue_.empty()) break;
            if (in_queue_.empty()) continue;
            pf = std::move(in_queue_.front());
            in_queue_.pop();
            lock.unlock();
            in_cv_.notify_one();
        }

        // 丢弃过期帧
        if (pf.wall_time_sec > 0) {
            int cur_fps = source_fps_;
            if (cur_fps <= 0) cur_fps = 25;
            double now_sec = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (now_sec - pf.wall_time_sec > 2.0 / cur_fps) {
                continue;
            }
        }

        auto t0 = std::chrono::steady_clock::now();
        std::vector<InferResult> results;
        ImageData bgr_image;
        bool ran_inference = false;
        int models_ran = 0;

        if (batch_scheduler_) {
            BatchRequest req;
            req.pipeline_id = cfg_.id;
            req.y_plane = const_cast<uint8_t*>(pf.y_ptr());
            req.uv_plane = const_cast<uint8_t*>(pf.uv_ptr());
            req.width = pf.width;
            req.height = pf.height;
            auto future = batch_scheduler_->submit(req);
            while (!future->ready) {
                std::this_thread::yield();
            }
            results = std::move(future->results);
            bgr_image = std::move(future->bgr_image);
            ran_inference = !results.empty();
        } else {
            models_ran = infer_group_->run_models(
                const_cast<uint8_t*>(pf.y_ptr()),
                const_cast<uint8_t*>(pf.uv_ptr()),
                pf.width, pf.height, pf.width, pf.width,
                &results, &bgr_image);
            ran_inference = models_ran > 0;
        }
        auto t1 = std::chrono::steady_clock::now();

        // 更新缓存：有检测结果时更新，无检测结果时清空
        if (!results.empty()) {
            cached_results_ = results;
            has_cached_results = true;
        } else if (ran_inference) {
            // 推理跑了但无结果 → 清空缓存，后续帧不再画旧框
            cached_results_.clear();
            has_cached_results = false;
        }

        // 非预览路：跳过绘制和编码
        if (!cfg_.enable_preview) {
            if (!bgr_image.empty() && ++snapshot_counter % snapshot_interval_ == 0) {
                std::lock_guard<std::mutex> lock(snapshot_mtx_);
                latest_bgr_ = std::make_shared<ImageData>(bgr_image);
            }
            stats_.record_frame(last_decode_us_.load(), 0, 0, 0);
            continue;
        }

        // 预览路：始终绘制（首次推理前无缓存也不画框，首次推理后有缓存则复用）
        if (!bgr_image.empty()) {
            if (!results.empty()) {
                draw_engine_->draw(bgr_image, results);
            } else if (has_cached_results) {
                draw_engine_->draw(bgr_image, cached_results_);
            }
        }
        auto t2 = std::chrono::steady_clock::now();
        if (bgr_image.empty()) continue;

        if (++snapshot_counter % snapshot_interval_ == 0) {
            std::lock_guard<std::mutex> lock(snapshot_mtx_);
            latest_bgr_ = std::make_shared<ImageData>(bgr_image);
        }

        EncodedFrame ef;
        ef.bgr_image = std::move(bgr_image);
        ef.width = pf.width;
        ef.height = pf.height;
        ef.pts = pf.pts;

        {
            std::unique_lock<std::mutex> lock(out_mtx_);
            if (out_queue_.size() >= out_max_size_) {
                out_queue_.pop();
            }
            out_queue_.push(std::move(ef));
            lock.unlock();
            out_cv_.notify_one();
        }

        int64_t infer_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        int64_t draw_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        stats_.record_frame(last_decode_us_.load(), infer_us, draw_us, last_encode_us_.load());
        t_last = t2;
    }
}

// ── 编码段（编码线程） ──

void Pipeline::encode_loop() {
    while (running_.load() || !out_queue_.empty()) {
        // 推流永久失败（地址被占用等）→ 仅设置错误信息，不改 running_
        // 由前端检测 init_error 后调用 /stop 接口统一停止
        if (encoder_ && encoder_->has_permanently_failed()) {
            init_error_ = encoder_->last_error();
            std::cerr << "[Pipeline:" << cfg_.id << "] Encoder permanently failed: "
                      << encoder_->last_error() << std::endl;
            break;
        }

        EncodedFrame ef;
        {
            std::unique_lock<std::mutex> lock(out_mtx_);
            out_cv_.wait_for(lock, std::chrono::milliseconds(100),
                [this]() { return !out_queue_.empty() || !running_.load(); });
            if (out_queue_.empty()) {
                if (!running_.load()) break;
                continue;
            }
            ef = std::move(out_queue_.front());
            out_queue_.pop();
        }

        if (!encoder_opened_.load()) {
            if (!encoder_->open(cfg_.output_url, ef.width, ef.height, source_fps_)) {
                if (encoder_->has_permanently_failed()) {
                    init_error_ = encoder_->last_error();
                    std::cerr << "[Pipeline:" << cfg_.id << "] " << init_error_ << std::endl;
                    break;
                }
                static auto last_enc_err_ = std::chrono::steady_clock::now();
                auto now = std::chrono::steady_clock::now();
                if (now - last_enc_err_ > std::chrono::seconds(5)) {
                    std::cerr << "[Pipeline:" << cfg_.id << "] Encoder open failed: " << cfg_.output_url << std::endl;
                    last_enc_err_ = now;
                }
                continue;
            }
            encoder_opened_ = true;
        }

        int64_t enc_us = encoder_->encode_timed(ef.bgr_image);
        last_encode_us_ = enc_us;
    }
}

// ── 模型动态管理 ──

bool Pipeline::update_config(const TaskConfig& cfg) {
    cfg_.enable_preview = cfg.enable_preview;
    cfg_.input_url = cfg.input_url;
    cfg_.output_url = cfg.output_url;
    cfg_.preview_url = cfg.preview_url;
    cfg_.decoder = cfg.decoder;
    cfg_.encoder = cfg.encoder;
    cfg_.draw = cfg.draw;
    cfg_.models = cfg.models; // 模型列表由调用方（update_task）通过 diff 同步
    cfg_.name = cfg.name;
    std::cout << "[Pipeline] Config updated: " << cfg_.id
              << " enable_preview=" << cfg_.enable_preview
              << " input=" << cfg_.input_url << std::endl;
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

bool Pipeline::latest_jpeg(std::vector<uint8_t>* out, int quality) {
    if (!out) return false;
    std::shared_ptr<ImageData> snap;
    {
        std::lock_guard<std::mutex> lock(snapshot_mtx_);
        if (!latest_bgr_ || latest_bgr_->empty()) return false;
        snap = latest_bgr_;
    }
    cv::Mat mat;
    snap->to_mat(mat, true);
    if (mat.empty()) return false;
    cv::Mat bgr;
    if (mat.channels() == 4) cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
    else if (mat.channels() == 1) cv::cvtColor(mat, bgr, cv::COLOR_GRAY2BGR);
    else bgr = mat;
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, quality };
    return cv::imencode(".jpg", bgr, *out, params);
}
