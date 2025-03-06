//
// Created by aichao on 2025/2/20.
//
#include "allocate.h"
#include <memory>

namespace modeldeploy {
    bool MDHostAllocator::operator()(void** ptr, size_t size) const {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }

    void MDHostFree::operator()(void* ptr) const { free(ptr); }
}
