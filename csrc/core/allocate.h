//
// Created by aichao on 2025/2/20.
//
#pragma once

namespace modeldeploy {
    class MDHostAllocator {
    public:
        bool operator()(void** ptr, size_t size) const;
    };

    class MDHostFree {
    public:
        void operator()(void* ptr) const;
    };
}
