#pragma once
#include <string>

namespace modeldeploy {
    struct NcnnBackendOption {
        int device_id = 0;
        int cpu_thread_num = -1;
        bool model_from_memory = false;
        std::string param_buffer;
        std::string bin_buffer;
    };
} // namespace modeldeploy
