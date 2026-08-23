#include "csrc/pipeline/edge.h"

namespace modeldeploy::pipeline {

bool edge_type_compatible(const Edge& edge) {
    return edge.src_type == edge.dst_type;
}

} // namespace modeldeploy::pipeline
