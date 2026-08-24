#pragma once
#include <memory>
#include <string>
#include "core/md_decl.h"

namespace modeldeploy::audio::tool {
// ITN 后端选择：Lightweight=内置轻量实现（零依赖，默认）；WeText=WeTextProcessing
// （覆盖最全，需 OpenFst，仅在 ENABLE_WETEXT=ON 且依赖就绪时可用）。
enum class MODELDEPLOY_CXX_EXPORT ItnBackend { Lightweight, WeText };

// 统一的 ITN 入口：内部按后端路由到轻量或 WeTextProcessing 实现。
class MODELDEPLOY_CXX_EXPORT ItnEngine {
public:
    explicit ItnEngine(ItnBackend backend = ItnBackend::Lightweight);
    ~ItnEngine();
    ItnEngine(ItnEngine&&) noexcept;
    ItnEngine& operator=(ItnEngine&&) noexcept;
    ItnEngine(const ItnEngine&) = delete;
    ItnEngine& operator=(const ItnEngine&) = delete;

    // 口读 -> 书面
    std::string normalize(const std::string& text) const;
    [[nodiscard]] ItnBackend backend() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace modeldeploy::audio::tool
