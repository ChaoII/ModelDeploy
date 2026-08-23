#pragma once

#include "core/md_decl.h"
#include <string>

namespace modeldeploy::pipeline {

// 一条有向边：src 节点 out_port -> dst 节点 in_port，记录两端端口 type 供一致性校验。
struct Edge {
    std::string src_node;
    std::string src_port;
    std::string src_type;
    std::string dst_node;
    std::string dst_port;
    std::string dst_type;
};

// src_type 与 dst_type 一致才算类型兼容
MODELDEPLOY_CXX_EXPORT bool edge_type_compatible(const Edge& edge);

} // namespace modeldeploy::pipeline
