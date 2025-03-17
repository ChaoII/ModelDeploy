//
// Created by aichao on 2025/2/20.
//

#include <memory>
#include <cstdlib>
#include "csrc/core/allocate.h"

namespace modeldeploy {
    bool MDHostAllocator::operator()(void** ptr, const size_t size) const {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }

    void MDHostFree::operator()(void* ptr) const { free(ptr); }
}
