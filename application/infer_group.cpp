#include "infer_group.hpp"
#include <algorithm>
#include <iostream>

using namespace modeldeploy::vision;

namespace {
// 计算有效像素 ROI：优先 roi_norm（归一化，按帧宽高换算），否则用像素 roi
struct RoiRect { float x, y, w, h; bool valid; };
RoiRect effective_roi(const ModelConfig& mc, const ImageData& frame) {
    if (mc.roi_norm.size() >= 4 && mc.roi_norm[2] > 0.f && mc.roi_norm[3] > 0.f &&
        frame.width() > 0 && frame.height() > 0) {
        const float W = static_cast<float>(frame.width());
        const float H = static_cast<float>(frame.height());
        float x = std::clamp(mc.roi_norm[0], 0.f, 1.f) * W;
        float y = std::clamp(mc.roi_norm[1], 0.f, 1.f) * H;
        float w = std::clamp(mc.roi_norm[2], 0.f, 1.f) * W;
        float h = std::clamp(mc.roi_norm[3], 0.f, 1.f) * H;
        // 防御性裁剪到帧内，避免越界 rect 造成 crop 后偏移错位
        if (x > W) x = W; if (y > H) y = H;
        if (x + w > W) w = W - x;
        if (y + h > H) h = H - y;
        if (w <= 0.f || h <= 0.f) return {0, 0, 0, 0, false};
        return {x, y, w, h, true};
    }
    if (mc.roi.size() >= 4 && mc.roi[2] > 0 && mc.roi[3] > 0)
        return {static_cast<float>(mc.roi[0]), static_cast<float>(mc.roi[1]),
                static_cast<float>(mc.roi[2]), static_cast<float>(mc.roi[3]), true};
    return {0, 0, 0, 0, false};
}
}

// 按配置构建引擎：优先用 factory（须返回已加载引擎），否则回退到 InferenceEngine::load。
// 在锁外调用（可能耗时）；失败返回 nullptr（调用方按 fail-safe 处理）。
std::shared_ptr<InferenceEngine> InferGroup::build_engine(const ModelConfig& mcfg,
                                                          ModelFactory& factory) {
    std::shared_ptr<InferenceEngine> engine;
    if (factory) {
        engine = factory(mcfg);
        if (engine && !engine->is_loaded()) engine.reset();
    }
    if (!engine) {
        auto e = std::make_shared<InferenceEngine>();
        if (!e->load(mcfg)) return nullptr;   // 加载失败：不改动快照
        engine = std::move(e);
    }
    return engine;
}

bool InferGroup::load_models(const std::vector<ModelConfig>& mcfgs, ModelFactory factory) {
    // 一次性构建全部引擎，最后整体换快照（避免逐次 add 触发多次复制）
    std::vector<Entry> built;
    built.reserve(mcfgs.size());
    bool all_ok = true;
    for (const auto& mcfg : mcfgs) {
        auto engine = build_engine(mcfg, factory);
        if (!engine) {
            std::cerr << "[InferGroup] Failed to load model: " << mcfg.name << std::endl;
            all_ok = false;
            continue;
        }
        built.push_back(Entry{std::move(engine), std::make_shared<EntryState>()});
    }
    {
        std::lock_guard<std::mutex> lk(entries_mtx_);
        entries_ = std::make_shared<const std::vector<Entry>>(std::move(built));
    }
    return all_ok;
}

bool InferGroup::add_model(const ModelConfig& mcfg, ModelFactory factory) {
    auto engine = build_engine(mcfg, factory);   // 锁外构建
    if (!engine) return false;
    std::lock_guard<std::mutex> lk(entries_mtx_);
    auto next = std::make_shared<std::vector<Entry>>(*entries_);   // 写时复制
    next->push_back(Entry{std::move(engine), std::make_shared<EntryState>()});
    entries_ = std::move(next);                  // 原子换指针
    return true;
}

bool InferGroup::remove_model(const std::string& name) {
    std::lock_guard<std::mutex> lk(entries_mtx_);
    auto next = std::make_shared<std::vector<Entry>>(*entries_);
    auto it = std::find_if(next->begin(), next->end(), [&](const Entry& e) {
        return e.engine && e.engine->config().name == name;
    });
    if (it == next->end()) return false;
    next->erase(it);
    entries_ = std::move(next);
    return true;
}

bool InferGroup::update_model(const ModelConfig& mcfg, ModelFactory factory) {
    // 锁外构建新引擎（可能耗时/下载），且只构建一次：换入的必须就是这里校验过的同一实例
    // （无二次加载、无丢弃）。factory 契约是"返回可用引擎或 nullptr"，但生产 factory
    // PipelineManager::create_engine 对 classification/pedestrian_attribute/ocr/lpr 在加载
    // 失败时仍返回非空引擎（只调用 eng->load 不检查返回值），仅判空会把未就绪引擎换入。
    // 因此这里强制 is_loaded() 就绪校验并下沉到本层 —— 所有调用方（新控制面路由 /
    // 旧 http_server PATCH / 未来调用方）都被覆盖。未就绪即 fail-safe：不改快照，
    // 旧模型继续服务。
    // 注意：与 add_model/load_models（经 build_engine，factory 未就绪时会回退直连加载）不同，
    // update_model 在 factory 已提供时不再回退 InferenceEngine::load —— 避免对同一配置重复
    // 加载，并让失败原因保持在调用方 factory 上。factory 为空/返回 nullptr 一律 fail-safe。
    std::shared_ptr<InferenceEngine> engine;
    if (factory) {
        engine = factory(mcfg);
        if (engine && !engine->is_loaded()) engine.reset();   // 未就绪 → 拒绝换入
    }
    if (!engine) return false;

    std::lock_guard<std::mutex> lk(entries_mtx_);
    auto next = std::make_shared<std::vector<Entry>>(*entries_);   // 复制快照
    bool found = false;
    for (auto& e : *next) {
        if (e.engine && e.engine->config().name == mcfg.name) {
            e.engine = std::move(engine);   // 仅替换目标引擎；state 指针复用（不重置跳帧计数）
            found = true;
            break;
        }
    }
    if (!found) next->push_back(Entry{std::move(engine), std::make_shared<EntryState>()});
    entries_ = std::move(next);             // 原子换指针（写时复制）
    return true;
}

void InferGroup::clear() {
    std::lock_guard<std::mutex> lk(entries_mtx_);
    entries_ = std::make_shared<const std::vector<Entry>>();
}

bool InferGroup::empty() const {
    std::lock_guard<std::mutex> lk(entries_mtx_);
    return entries_->empty();
}

bool InferGroup::run_models(
    const ImageData& frame,
    std::vector<std::pair<std::string, std::vector<DetectionResult>>>* sdk_dets,
    std::vector<std::pair<std::string, InferResult>>* non_det) {
    // 入口取一次不可变快照：shared_ptr 拷贝保证整段推理期间引擎与状态存活；
    // 期间并发的增删改只会换出新快照，不影响本次只读遍历（修迭代器失效/数据竞争）。
    std::shared_ptr<const std::vector<Entry>> snap;
    {
        std::lock_guard<std::mutex> lk(entries_mtx_);
        snap = entries_;
    }
    if (sdk_dets) sdk_dets->clear();
    if (non_det) non_det->clear();
    bool any = false;
    for (const auto& en : *snap) {
        auto* e = en.engine.get();
        auto& st = *en.state;                  // per-entry 状态：仅 detect 线程写
        const auto& mc = e->config();
        const int interval = mc.interval > 0 ? mc.interval : 1;
        // interval 抽帧：每 interval 帧推理一次，其余帧复用上次结果（无结果时强制先跑一次）
        const bool do_infer = !st.has_last || (st.frame_idx % interval == 0);
        const bool is_det = (mc.type == "detection" && e->det_model() != nullptr);

        if (do_infer) {
            if (is_det) {
                // ROI 区域裁剪推理：仅 CPU 帧（设备 NV12 不做裁剪）；结果按偏移回映射到全图
                const RoiRect rr = effective_roi(mc, frame);
                ImageData infer_frame = frame;
                float ox = 0.f, oy = 0.f;
                if (rr.valid && frame.device() == modeldeploy::Device::CPU) {
                    infer_frame = frame.crop({rr.x, rr.y, rr.w, rr.h});
                    ox = rr.x; oy = rr.y;
                }
                std::vector<DetectionResult> dets;
                if (e->predict_detection(infer_frame, &dets)) {
                    if (ox != 0.f || oy != 0.f)
                        for (auto& d : dets) { d.box.x += ox; d.box.y += oy; }
                    st.last_dets = std::move(dets);
                    st.has_last = true;
                }
            } else {
                // ROI：非 detection 族（如人脸）按区域裁剪，几何结果按偏移回映射；classification 无几何不做
                const RoiRect rr = effective_roi(mc, frame);
                const bool can_roi = rr.valid && frame.device() == modeldeploy::Device::CPU &&
                                     mc.type != "classification";
                ImageData infer_frame = frame;
                float ox = 0.f, oy = 0.f;
                if (can_roi) {
                    infer_frame = frame.crop({rr.x, rr.y, rr.w, rr.h});
                    ox = rr.x; oy = rr.y;
                }
                InferResult r;
                if (e->infer(infer_frame, &r)) {
                    if (ox != 0.f || oy != 0.f) {
                        for (auto& b : r.boxes) { b.x += ox; b.y += oy; }
                        for (auto& kps : r.keypoints)
                            for (auto& k : kps) { k.x += ox; k.y += oy; }
                    }
                    st.last_non_det = std::move(r);
                    st.has_last = true;
                }
            }
        }
        ++st.frame_idx;

        if (is_det) {
            if (sdk_dets) sdk_dets->push_back({mc.name, st.last_dets});
        } else {
            if (non_det) non_det->push_back({mc.name, st.last_non_det});
        }
        if (st.has_last) any = true;
    }
    return any;
}

modeldeploy::vision::detection::UltralyticsDet* InferGroup::det_model(const std::string& name) {
    // 取快照后即返回裸指针：指针归快照内引擎所有，并发热更新换出该条目的旧快照时可能悬垂
    // （详见头文件生命周期约束）。调用方须即取即用，绝不跨热更新/跨线程长期持有。
    std::shared_ptr<const std::vector<Entry>> snap;
    {
        std::lock_guard<std::mutex> lk(entries_mtx_);
        snap = entries_;
    }
    for (const auto& en : *snap) {
        if (en.engine && en.engine->config().name == name) return en.engine->det_model();
    }
    return nullptr;
}

const ModelConfig* InferGroup::config_of(const std::string& name) const {
    // 同上：返回快照内引擎的配置裸指针，并发热更新后可能悬垂（见头文件生命周期约束）。
    std::shared_ptr<const std::vector<Entry>> snap;
    {
        std::lock_guard<std::mutex> lk(entries_mtx_);
        snap = entries_;
    }
    for (const auto& en : *snap) {
        if (en.engine && en.engine->config().name == name) return &en.engine->config();
    }
    return nullptr;
}
