#include "pipeline.hpp"
#include <iostream>
#include <cstring>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace modeldeploy::vision;

Pipeline::Pipeline(TaskConfig cfg, ModelFactory factory)
    : cfg_(std::move(cfg)),
      model_factory_(std::move(factory)) {
}

std::string Pipeline::init_error() const {
    std::lock_guard<std::mutex> lock(init_error_mtx_);
    return init_error_;
}

void Pipeline::set_init_error(const std::string& msg) {
    std::lock_guard<std::mutex> lock(init_error_mtx_);
    init_error_ = msg;
}

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::start() {
    // 生命周期锁：防止 stop 在线程创建完成前 join（未 join 的 thread 析构会 std::terminate）
    std::lock_guard<std::mutex> lock(lifecycle_mtx_);
    // CAS：并发 start 只允许一个进入
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return true;
    // 上一轮线程已结束（否则 running_ 为 true，CAS 不会成功）；join 清理残留句柄
    if (detect_thread_.joinable()) detect_thread_.join();
    stopped_ = false;
    initialized_ = false;
    alive_token_ = std::make_shared<std::atomic<bool>>(true);
    set_init_error("");
    {
        std::lock_guard<std::mutex> dl(det_mtx_);
        det_queue_.clear();
    }

    // 模型加载：失败不阻断预览（预览可推原帧）
    infer_group_.load_models(cfg_.models, model_factory_);
    if (infer_group_.empty()) {
        std::cerr << "[Pipeline] " << cfg_.id << " no models loaded (preview only)" << std::endl;
    }
    draw_engine_ = std::make_unique<DrawEngine>(cfg_.draw);

    // 1) SDK 解码源（open 同步完成，之后回调异步）
    std::string err;
    if (!src_.open(cfg_.input_url, cfg_.decoder, &err)) {
        set_init_error("Decoder open failed: " + cfg_.input_url + " (" + err + ")");
        std::cerr << "[Pipeline] " << init_error() << std::endl;
        running_ = false;
        return true;
    }
    source_fps_ = src_.fps();
    if (source_fps_ <= 0) source_fps_ = 25;

    src_.set_callback([this, alive = alive_token_](modeldeploy::video::VideoFrame&& f) {
        // 任务已销毁/停止：跳过（解码线程可能在任务析构后在途调用此回调）
        if (!alive || !alive->load()) return;
        if (!running_.load()) return;
        this->push_detect_frame(std::move(f));
    });
    if (!src_.start(&err)) {
        set_init_error("Decoder start failed: " + err);
        std::cerr << "[Pipeline] " << init_error() << std::endl;
        src_.close();
        running_ = false;
        return true;
    }

    // 2) 预览编码段（GPU-direct 门控：源 CUDA 硬解设备帧 + 设备专用 + 硬编容器）
    if (cfg_.enable_preview && !cfg_.output_url.empty()) {
        const bool src_gpu = (cfg_.decoder.hw_accel == "cuda");
        const bool nvenc_codec = (cfg_.encoder.codec == "auto" || cfg_.encoder.codec == "h264_nvenc"
                                  || cfg_.encoder.codec == "nvh264enc");
        const bool gpu_direct = src_gpu && cfg_.decoder.device_only && nvenc_codec;
        if (!sink_.open(cfg_.output_url, src_.width(), src_.height(), source_fps_,
                        cfg_.encoder, gpu_direct, &err)) {
            set_init_error("Encoder open failed: " + cfg_.output_url + " (" + err + ")");
            std::cerr << "[Pipeline] " << init_error() << std::endl;
            src_.stop();
            running_ = false;
            return true;
        }
        sink_.start_async(&err);
    }

    // 3) 检测线程（单线程关键路径）
    detect_thread_ = std::thread([this]() {
        try {
            detect_loop();
        } catch (const std::exception& e) {
            set_init_error(e.what());
            std::cerr << "[Pipeline-detect] Fatal: " << e.what() << std::endl;
        } catch (...) {
            set_init_error("detect unknown error");
            std::cerr << "[Pipeline-detect] Fatal: unknown" << std::endl;
        }
        running_ = false;
        det_cv_.notify_all();
    });

    initialized_ = true;
    stats_.start();
    std::cout << "[Pipeline] Running: " << cfg_.id << " source_fps=" << source_fps_ << std::endl;
    return true;
}

void Pipeline::stop() {
    // 生命周期锁：与 start 串行化，保证线程已创建后再 join
    std::lock_guard<std::mutex> lock(lifecycle_mtx_);
    if (stopped_.load()) return;
    stopped_ = true;
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        running_ = false;
    }
    // 先停解码源（停止回调线程，不再入队），再唤醒检测线程退出
    src_.stop();
    det_cv_.notify_all();
    if (detect_thread_.joinable()) detect_thread_.join();
    sink_.stop_async();   // 排空待编码帧
    sink_.close();
    release_resources();
    stats_.print();
}

void Pipeline::release_resources() {
    // 先失效生命周期令牌：此后任何在途的解码回调都会跳过，不再触碰 this
    if (alive_token_) *alive_token_ = false;
    src_.close();
    sink_.close();
    infer_group_.clear();
    draw_engine_.reset();
    initialized_ = false;
    cached_results_.clear();

    {
        std::lock_guard<std::mutex> lock(det_mtx_);
        det_queue_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(snapshot_mtx_);
        latest_bgr_.reset();
    }
}

void Pipeline::push_detect_frame(modeldeploy::video::VideoFrame&& f) {
    std::lock_guard<std::mutex> lock(det_mtx_);
    // 有界队列满则丢最旧帧（保最新、控延迟）
    if (det_queue_.size() >= det_max_size_) det_queue_.pop_front();
    det_queue_.push_back(std::move(f));
    det_cv_.notify_one();
}

double Pipeline::model_threshold(const std::string& name) const {
    for (const auto& m : cfg_.models) {
        if (m.name == name) return m.confidence_threshold;
    }
    // 动态 add_model 加入的模型不在 cfg_.models：回退查 InferGroup 引擎配置
    const auto* mc = infer_group_.config_of(name);
    if (mc) return mc->confidence_threshold;
    return 0.5;
}

namespace {
// CPU packed BGR → 主机 NV12（重建自有缓冲的 ImageData，供编码路径）
modeldeploy::vision::ImageData bgr_to_nv12_host(const modeldeploy::vision::ImageData& bgr) {
    cv::Mat mat;
    if (!bgr.asMat(&mat)) return {};
    const int W = mat.cols, H = mat.rows;
    if (W <= 0 || H <= 0 || (W & 1) || (H & 1)) return {};
    cv::Mat i420;
    cv::cvtColor(mat, i420, cv::COLOR_BGR2YUV_I420);
    const size_t ysize = static_cast<size_t>(W) * static_cast<size_t>(H);
    const size_t usize = ysize / 4;
    const uint8_t* y = i420.data;
    const uint8_t* u = y + ysize;
    const uint8_t* v = u + usize;
    auto holder = std::make_shared<std::vector<uint8_t>>(ysize + ysize / 2);
    std::memcpy(holder->data(), y, ysize);
    uint8_t* uv = holder->data() + ysize;
    const int h2 = H / 2, w2 = W / 2;
    for (int r = 0; r < h2; ++r) {
        for (int c = 0; c < w2; ++c) {
            uv[r * W + 2 * c]     = u[r * w2 + c];
            uv[r * W + 2 * c + 1] = v[r * w2 + c];
        }
    }
    modeldeploy::vision::ImageData::Plane pl[2] = {
        {holder->data(), W},
        {holder->data() + ysize, W},
    };
    std::shared_ptr<void> owner(holder, holder->data());
    return modeldeploy::vision::ImageData::from_planes(
        pl, 2, MdImageType::NV12, W, H, modeldeploy::Device::CPU, owner);
}
}

void Pipeline::draw_non_det(ImageData& frame, const std::vector<InferResult>& results) {
    if (!draw_engine_ || results.empty() || frame.empty()) return;
    // host NV12：DrawEngine CPU draw()（vis_det/vis_keypoints 保留 face 关键点/标签格式）需
    // BGR packed → 转 BGR 标注后重建 NV12 交付编码（face/classification 呈现到输出帧）。
    if (frame.type() == MdImageType::NV12 && frame.plane_count() >= 2 &&
        frame.device() == modeldeploy::Device::CPU) {
        ImageData bgr = ImageData::cvt_color(frame, ColorConvertType::CVT_NV122PKG_BGR);
        if (bgr.empty()) return;
        draw_engine_->draw(bgr, results);
        ImageData nv12 = bgr_to_nv12_host(bgr);
        if (!nv12.empty()) frame = std::move(nv12);
        return;
    }
    // device NV12：draw_gpu 就地零拷贝（不破坏 GPU 直编 D2D）
    if (frame.type() == MdImageType::NV12 && frame.plane_count() >= 2) {
        draw_engine_->draw_gpu(frame, results, cfg_.draw.show_label, cfg_.draw.show_score);
        return;
    }
    // 其它（packed BGR 等）：CPU draw()
    draw_engine_->draw(frame, results);
}

void Pipeline::update_snapshot(const ImageData& frame, int64_t& counter) {
    if (++counter % snapshot_interval_ != 0) return;
    ImageData snap;
    if (frame.device() != modeldeploy::Device::CPU) {
        // 设备帧：toCpu 做深拷贝，脱离解码池复用缓冲的生命周期
        if (!frame.toCpu(&snap)) return;
    } else {
        // CPU 帧：toCpu 仅浅 clone 共享解码池缓冲，需深拷贝独立所有权
        snap = frame.clone();
    }
    std::lock_guard<std::mutex> lock(snapshot_mtx_);
    latest_bgr_ = std::make_shared<ImageData>(std::move(snap));
}

// ── 检测循环（应用单线程关键路径：detect + draw + encode_async） ──

void Pipeline::detect_loop() {
    int64_t snapshot_counter = 0;
    bool encode_failed_reported = false;
    auto t_last = std::chrono::steady_clock::now();
    while (running_.load()) {
        modeldeploy::video::VideoFrame f;
        {
            std::unique_lock<std::mutex> lock(det_mtx_);
            det_cv_.wait(lock, [this]() {
                return !det_queue_.empty() || !running_.load();
            });
            if (!running_.load() && det_queue_.empty()) break;
            if (det_queue_.empty()) continue;
            f = std::move(det_queue_.front());
            det_queue_.pop_front();
        }

        auto t0 = std::chrono::steady_clock::now();

        // 推理 + 绘制，均在 frame.image（设备 NV12）上零拷贝
        std::vector<std::pair<std::string, std::vector<DetectionResult>>> sdk_dets;
        std::vector<std::pair<std::string, InferResult>> non_det;
        infer_group_.run_models(f.image, &sdk_dets, &non_det);
        for (auto& [name, dets] : sdk_dets) {
            auto* det = infer_group_.det_model(name);
            if (det) det->draw_result(f.image, dets, model_threshold(name));
        }
        // 非 detection（face/classification）标注到输出帧（DrawEngine）
        if (!non_det.empty()) {
            std::vector<InferResult> res;
            res.reserve(non_det.size());
            for (auto& [name, r] : non_det) res.push_back(std::move(r));
            draw_non_det(f.image, res);
        }
        last_frame_pts_ = static_cast<int64_t>(f.pts_ms);

        auto t1 = std::chrono::steady_clock::now();

        // 预览编码（encode_async；GPU 直编 D2D）
        if (cfg_.enable_preview && !cfg_.output_url.empty()) {
            if (!sink_.encode(f.image) || sink_.has_failed()) {
                if (!encode_failed_reported) {
                    encode_failed_reported = true;
                    set_init_error("encode failed: " + sink_.last_error());
                }
            }
        }

        auto t2 = std::chrono::steady_clock::now();

        // 低频快照（不占每帧关键路径）
        update_snapshot(f.image, snapshot_counter);

        auto t3 = std::chrono::steady_clock::now();
        int64_t infer_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        int64_t draw_us  = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        int64_t enc_us   = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        stats_.record_frame(0, infer_us, draw_us, enc_us);

        // 摄入 SDK 编解码统计（轻量：src_/sink_ 已聚合的标量拷贝；解码侧帧率快照）
        {
            const auto sst = src_.stats();
            const auto kst = sink_.stats();
            stats_.ingest_sdk(sst.frames_in, sst.frames_out, sst.dropped,
                              sst.avg_decode_ms, sst.reconnect_count,
                              kst.avg_encode_ms);
        }
        t_last = t3;
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

void Pipeline::update_preview_mode(bool enable) {
    cfg_.enable_preview = enable;
}

bool Pipeline::add_model(const ModelConfig& mcfg) {
    return infer_group_.add_model(mcfg, model_factory_);
}

bool Pipeline::remove_model(const std::string& name) {
    return infer_group_.remove_model(name);
}

bool Pipeline::update_model(const std::string& name, const ModelConfig& mcfg) {
    // 新 InferGroup 无原位 update：先删旧、再加新（等价语义）
    infer_group_.remove_model(name);
    return infer_group_.add_model(mcfg, model_factory_);
}

bool Pipeline::latest_bgr_snapshot(std::shared_ptr<ImageData>* out) const {
    if (!out) return false;
    std::lock_guard<std::mutex> lock(snapshot_mtx_);
    if (!latest_bgr_ || latest_bgr_->empty()) return false;
    *out = latest_bgr_;   // shared_ptr 拷贝：快照独立于 pipeline 生命周期
    return true;
}

bool Pipeline::encode_jpeg(const std::shared_ptr<ImageData>& snap,
                           std::vector<uint8_t>* out, int quality) {
    if (!out || !snap) return false;
    cv::Mat mat;
    ImageData cpu;
    if (!snap->toCpu(&cpu)) return false;
    // NV12 快照（设备或 host）→ BGR，再走 JPEG 编码
    if (cpu.type() == MdImageType::NV12) {
        cpu = ImageData::cvt_color(cpu, ColorConvertType::CVT_NV122PKG_BGR);
    }
    (void)cpu.asMat(&mat);
    if (mat.empty()) return false;
    cv::Mat bgr;
    if (mat.channels() == 4) cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
    else if (mat.channels() == 1) cv::cvtColor(mat, bgr, cv::COLOR_GRAY2BGR);
    else bgr = mat;
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, quality };
    return cv::imencode(".jpg", bgr, *out, params);
}

bool Pipeline::latest_jpeg(std::vector<uint8_t>* out, int quality) {
    std::shared_ptr<ImageData> snap;
    if (!latest_bgr_snapshot(&snap)) return false;
    return encode_jpeg(snap, out, quality);
}
