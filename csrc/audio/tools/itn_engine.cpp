#include "audio/tools/itn_engine.h"
#include "audio/tools/itn.h"
#include <cstdlib>

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
        // 模型目录由调用方通过环境变量 MODELDEPLOY_WETEXT_DIR 或固定相对路径给出。
        // 需要 tagger.fst 与 verbalizer.fst（由 WeTextProcessing 语法编译生成）。
        const char* dir = std::getenv("MODELDEPLOY_WETEXT_DIR");
        if (dir && *dir) {
            impl_->wn = std::make_shared<wetext::Processor>(
                std::string(dir) + "/tagger.fst",
                std::string(dir) + "/verbalizer.fst");
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
        if (impl_->ready && impl_->wn) return impl_->wn->Normalize(text);
        // 依赖缺失/未加载时退化到轻量实现
        return impl_->lite.normalize(text);
    }
#endif
    return impl_->lite.normalize(text);
}

ItnBackend ItnEngine::backend() const { return impl_->backend; }

} // namespace modeldeploy::audio::tool
