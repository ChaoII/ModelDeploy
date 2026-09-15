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
    /// 生命周期约束（悬垂风险，调用方必读）：返回的裸指针指向**当前 COW 快照**内的对象，
    /// 仅在该同名条目未被换出、且其快照仍被持有期间有效。并发的 update_model / remove_model /
    /// clear 会原子换出新快照；当旧快照失去最后一个持有者时旧引擎被释放，先前取得的裸指针
    /// 立即失效（再解引用即访问已释放内存 = UB）。调用方必须：即取即用、只在同步逻辑内使用、
    /// 绝不跨热更新或跨线程长期持有。若需跨热更新保活，应改为持有 shared_ptr<InferenceEngine>。
    modeldeploy::vision::detection::UltralyticsDet* det_model(const std::string& name);
    /// 按模型名取 ModelConfig（含动态 add_model 加入的模型），未找到返回 nullptr。
    /// 生命周期约束同 det_model：裸指针指向快照内引擎，并发热更新后可能悬垂，须即取即用。
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
