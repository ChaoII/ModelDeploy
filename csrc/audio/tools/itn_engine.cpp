#include "audio/tools/itn_engine.h"
#include "audio/tools/itn.h"
#include "core/md_log.h"
#include <cstdlib>
#include <filesystem>

#ifdef MODELDEPLOY_HAS_WETEXT
#include <memory>
#include <string>
// 接口对齐 wenet-e2e/WeTextProcessing runtime（依赖 OpenFst）：
//   wetext::Processor(const std::string& tagger_fst, const std::string& verbalizer_fst);
//   std::string Normalize(const std::string& input);
#include "processor/wetext_processor.h" // IWYU pragma: keep
#endif

namespace modeldeploy::audio::tool {

struct ItnEngine::Impl {
    explicit Impl(ItnBackend b) : backend(b) {}
    ItnBackend backend = ItnBackend::Lightweight;
    InverseTextNormalizer lite;
#ifdef MODELDEPLOY_HAS_WETEXT
    std::shared_ptr<wetext::Processor> wn;
    bool ready = false;
#endif
};

ItnEngine::ItnEngine(ItnBackend backend)
    : impl_(std::make_unique<Impl>(backend)) {
#ifdef MODELDEPLOY_HAS_WETEXT
    if (backend == ItnBackend::WeText) {
        // 模型目录由调用方通过环境变量 MODELDEPLOY_WETEXT_DIR 给出。
        // 优先 wenet 命名约定 zh_itn_tagger.fst / zh_itn_verbalizer.fst
        //（文件名必须包含 "zh_itn_" 等方向前缀，wetext::Processor 据此判定 TN/ITN/语言；
        //  否则真实 glog 下构造会 LOG(FATAL)）。
        // 兼容旧约定 tagger.fst / verbalizer.fst（此时需调用方保证前缀合法）。
        const char* dir = std::getenv("MODELDEPLOY_WETEXT_DIR");
        if (dir && *dir) {
            namespace fs = std::filesystem;
            const fs::path base(dir);
            std::string tagger, verbalizer;
            if (fs::exists(base / "zh_itn_tagger.fst") &&
                fs::exists(base / "zh_itn_verbalizer.fst")) {
                tagger = (base / "zh_itn_tagger.fst").string();
                verbalizer = (base / "zh_itn_verbalizer.fst").string();
            } else if (fs::exists(base / "tagger.fst") &&
                       fs::exists(base / "verbalizer.fst")) {
                tagger = (base / "tagger.fst").string();
                verbalizer = (base / "verbalizer.fst").string();
            } else {
                MD_LOG_ERROR << "[ItnEngine] WeText 模型缺失（需 zh_itn_tagger.fst + zh_itn_verbalizer.fst 或 tagger.fst + verbalizer.fst 于 "
                             << dir << "），退化到轻量实现。" << std::endl;
                return;
            }
            impl_->wn = std::make_shared<wetext::Processor>(tagger, verbalizer);
            impl_->ready = true;
        }
    }
#else
    (void)backend;
#endif
}

ItnEngine::~ItnEngine() = default;
ItnEngine::ItnEngine(ItnEngine&&) noexcept = default;
ItnEngine& ItnEngine::operator=(ItnEngine&&) noexcept = default;

std::string ItnEngine::normalize(const std::string& text) const {
#ifdef MODELDEPLOY_HAS_WETEXT
    if (impl_->backend == ItnBackend::WeText) {
        if (impl_->ready && impl_->wn) {
            try {
                return impl_->wn->Normalize(text);
            } catch (const std::exception& e) {
                // weText runtime 失败（模型/语法不匹配等）时 fail-soft 到轻量实现，避免异常外泄
                MD_LOG_WARN << "[ItnEngine] WeText normalize 失败，退化到轻量实现: " << e.what() << std::endl;
                return impl_->lite.normalize(text);
            }
        }
        // 依赖缺失/未加载时退化到轻量实现
        return impl_->lite.normalize(text);
    }
#endif
    return impl_->lite.normalize(text);
}

ItnBackend ItnEngine::backend() const { return impl_->backend; }

} // namespace modeldeploy::audio::tool
