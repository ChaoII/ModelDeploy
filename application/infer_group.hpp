#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "config.hpp"
#include "inference_engine.hpp"
#include "csrc/vision/common/image_data.h"
#include "csrc/vision/common/result.h"

/// 多模型调度组：管理同一路视频上的多个推理模型（纯模型薄封装）
/// 每帧串行跑各模型；detection 模型额外缓存 SDK DetectionResult 供 pipeline 设备绘制
/// 支持 per-model interval（每 N 帧推理一次，其余帧复用上次结果，省算力且不闪烁）
class InferGroup {
public:
    using ModelFactory = std::function<std::unique_ptr<InferenceEngine>(const ModelConfig&)>;
    bool load_models(const std::vector<ModelConfig>& mcfgs, ModelFactory factory);
    bool add_model(const ModelConfig& mcfg, ModelFactory factory);
    bool remove_model(const std::string& name);
    /// 热更新：按 mcfg.name 替换对应模型（同名不存在则追加）。
    /// 新引擎在锁外构建（可能耗时），只构建一次：factory 契约是"返回已加载引擎或 nullptr"，
    /// 为空/返回 nullptr 时返回 false（不回退 InferenceEngine::load，避免重复加载）。
    /// **换入前强制就绪校验**（InferenceEngine::is_loaded()）：未就绪（含 factory 返回的
    /// 非空未加载引擎，如 create_engine 对 cls/ocr/lpr 加载失败）一律返回 false 且不改动
    /// 快照（fail-safe，旧模型继续服务）。校验下沉到本层，保证所有调用方
    /// （新控制面路由 / 旧 PATCH / 未来调用方）都被覆盖；换入的即校验过的同一实例。
    bool update_model(const ModelConfig& mcfg, ModelFactory factory);
    void clear();
    bool empty() const;
    /// 对一帧依次跑全部模型；sdk_dets 收集各 detection 模型的 SDK DetectionResult（供设备绘制）；
    /// non_det 收集各非 detection 模型（face/classification）的 InferResult（供 DrawEngine 标注，可空）
    /// 线程安全：入口一次性取不可变快照，整段只读；期间并发增删改不会影响本次推理。
    bool run_models(const modeldeploy::vision::ImageData& frame,
                    std::vector<std::pair<std::string, std::vector<modeldeploy::vision::DetectionResult>>>* sdk_dets,
                    std::vector<std::pair<std::string, InferResult>>* non_det = nullptr);

    /// 模型句柄：**强引用**持有快照内引擎，句柄存活期间 config()/det_model() 始终有效。
    /// 用于消除下面裸指针访问器的悬垂风险（并发审计 HIGH：并发热更新换出旧快照后，
    /// 原裸指针立即失效 → UAF）。调用方直接按值持有本句柄即可安全跨热更新使用，
    /// 无需关心快照生命周期；句柄析构时若该引擎已不在当前快照中则随之释放。
    class ModelRef {
    public:
        ModelRef() = default;
        explicit ModelRef(std::shared_ptr<InferenceEngine> engine) : engine_(std::move(engine)) {}

        explicit operator bool() const { return static_cast<bool>(engine_); }
        const ModelConfig* config() const { return engine_ ? &engine_->config() : nullptr; }
        modeldeploy::vision::detection::UltralyticsDet* det_model() const {
            return engine_ ? engine_->det_model() : nullptr;
        }
        const std::shared_ptr<InferenceEngine>& engine() const { return engine_; }

    private:
        std::shared_ptr<InferenceEngine> engine_;
    };

    /// 按模型名取强引用句柄（并发安全，供生产路径使用）；未找到返回空句柄。
    ModelRef find_model(const std::string& name) const;

    /// ⚠️ 以下两个裸指针访问器仅为诊断/测试保留（非生产路径）：返回的快照内对象在
    /// 并发热更新换出同名条目后立即悬垂，**不得**在 pipeline 等生产路径使用。
    /// 生产代码请改用 find_model() 取得 ModelRef。
    modeldeploy::vision::detection::UltralyticsDet* det_model(const std::string& name);
    /// 按模型名取 ModelConfig（含动态 add_model 加入的模型），未找到返回 nullptr。
    /// 生命周期约束同 det_model（悬垂风险），生产路径请改用 find_model()。
    const ModelConfig* config_of(const std::string& name) const;

private:
    /// per-entry 可变状态：仅 detect 线程访问；热交换时代码复用（不重置跳帧计数）
    struct EntryState {
        int64_t frame_idx = 0;                 // 已处理帧数（用于 interval 抽帧）
        bool has_last = false;                 // 是否已有可复用的上次结果
        std::vector<modeldeploy::vision::DetectionResult> last_dets;   // detection 缓存
        InferResult last_non_det;              // 非 detection 缓存
    };
    struct Entry {
        std::shared_ptr<InferenceEngine> engine;
        std::shared_ptr<EntryState> state;    // 副本间复用（替换模型不重置跳帧计数）
    };
    // 不可变快照：run_models 只读；增删改在副本上完成后原子替换指针（写时复制）。
    // entries_mtx_ 仅保护"读指针 / 复制 / 换指针"，绝不在锁内做模型加载等耗时操作。
    std::shared_ptr<const std::vector<Entry>> entries_ =
        std::make_shared<const std::vector<Entry>>();
    mutable std::mutex entries_mtx_;

    /// 在锁外构建引擎（可能耗时）：优先 factory（须返回已加载引擎），否则回退 InferenceEngine::load。
    /// 失败返回 nullptr，由调用方按 fail-safe 处理（不改快照）。
    std::shared_ptr<InferenceEngine> build_engine(const ModelConfig& mcfg, ModelFactory& factory);
};
