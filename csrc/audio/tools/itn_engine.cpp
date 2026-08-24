#include "audio/tools/itn_engine.h"
#include "audio/tools/itn.h"
#include <cstdlib>

#ifdef MODELDEPLOY_HAS_WETEXT
#include <memory>
#include <string>
// 由 ENABLE_WETEXT 提供 include 路径；接口对齐 WeTextProcessing/sherpa-onnx：
//   CnItnProcessor(const std::string& tagger_fst, const std::string& verbalizer_fst);
//   std::string normalize(const std::string& input);
#include "we-text-processing/csrc/cn_itn_processor.h" // IWYU pragma: keep
#endif

namespace modeldeploy::audio::tool {

struct ItnEngine::Impl {
    explicit Impl(ItnBackend b) : backend(b) {}
    ItnBackend backend = ItnBackend::Lightweight;
    InverseTextNormalizer lite;
#ifdef MODELDEPLOY_HAS_WETEXT
    std::shared_ptr<CnItnProcessor> cn;
    bool ready = false;
#endif
};

ItnEngine::ItnEngine(ItnBackend backend)
    : impl_(std::make_unique<Impl>(backend)) {
#ifdef MODELDEPLOY_HAS_WETEXT
    if (backend == ItnBackend::WeText) {
        // 模型目录由调用方通过环境变量 MODELDEPLOY_WETEXT_DIR 或固定相对路径给出。
        // 需要 tagger.fst 与 verbalizer.fst。
        const char* dir = std::getenv("MODELDEPLOY_WETEXT_DIR");
        if (dir && *dir) {
            impl_->cn = std::make_shared<CnItnProcessor>(
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
        if (impl_->ready && impl_->cn) return impl_->cn->normalize(text);
        // 依赖缺失/未加载时退化到轻量实现
        return impl_->lite.normalize(text);
    }
#endif
    return impl_->lite.normalize(text);
}

ItnBackend ItnEngine::backend() const { return impl_->backend; }

} // namespace modeldeploy::audio::tool
