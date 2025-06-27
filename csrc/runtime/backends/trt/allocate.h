#pragma once

#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>

#include "csrc/utils/utils.h"

namespace modeldeploy {
    class MODELDEPLOY_CXX_EXPORT FDHostAllocator {
    public:
        bool operator()(void** ptr, size_t size) const;
    };

    class MODELDEPLOY_CXX_EXPORT FDHostFree {
    public:
        void operator()(void* ptr) const;
    };

#ifdef WITH_GPU

    class MODELDEPLOY_CXX_EXPORT FDDeviceAllocator {
    public:
        bool operator()(void** ptr, size_t size) const;
    };

    class MODELDEPLOY_CXX_EXPORT FDDeviceFree {
    public:
        void operator()(void* ptr) const;
    };

    class MODELDEPLOY_CXX_EXPORT FDDeviceHostAllocator {
    public:
        bool operator()(void** ptr, size_t size) const;
    };

    class MODELDEPLOY_CXX_EXPORT FDDeviceHostFree {
    public:
        void operator()(void* ptr) const;
    };

#endif
}
