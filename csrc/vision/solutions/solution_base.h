#pragma once
#include "core/md_decl.h"
namespace modeldeploy::vision::solution {
struct MODELDEPLOY_CXX_EXPORT SolutionBase {
    virtual ~SolutionBase() = default;
    virtual void reset() = 0;
};
} // namespace modeldeploy::vision::solution
