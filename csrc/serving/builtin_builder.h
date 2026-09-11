//
// ServingServer 内置通用 HandleBuilder —— 按 manifest 的 type 构造常见模型族。
//
// 覆盖 det/cls/seg/pose/obb/sem/depth/face/ocr/lpr；构造失败/未初始化时返回仅含元数据的
// 句柄（infer 为空），由 ModelRepo::load 统一置 Failed，绝不因单个模型崩溃进程。
// 仅 BUILD_VISION=ON 时编入真实实现；否则返回明确错误的占位 builder。
//
#pragma once

#include "core/md_decl.h"
#include "serving/config.h"
#include "serving/model_repo.h"

namespace modeldeploy::serving {

// 返回内置 builder（使用 cfg.font_path 作可视化字体）。用户可自行注入 builder 覆盖。
MODELDEPLOY_CXX_EXPORT HandleBuilder make_builtin_builder(const ServingConfig& cfg);

}  // namespace modeldeploy::serving
