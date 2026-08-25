#include "core/device_validate.h"

#ifdef WITH_GPU
#include <cuda_runtime.h>
#endif

namespace modeldeploy {

bool validate_pointer_device(const void* ptr, const Device dev, const int device_id,
                             DeviceValidateCode* out_code) {
    const auto set = [&](DeviceValidateCode c) { if (out_code) *out_code = c; };
    switch (dev) {
        case Device::CPU:
            if (!ptr) { set(DeviceValidateCode::Invalid); return false; }
            set(DeviceValidateCode::Ok); return true;
        case Device::GPU:
#ifdef WITH_GPU
        {
            if (!ptr) { set(DeviceValidateCode::Invalid); return false; }
            cudaPointerAttributes attr;
            if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess ||
                attr.type != cudaMemoryTypeDevice) {
                set(DeviceValidateCode::Invalid); return false;
            }
            if (attr.device != device_id) {
                set(DeviceValidateCode::Invalid); return false;
            }
            set(DeviceValidateCode::Ok); return true;
        }
#else
            set(DeviceValidateCode::Unsupported); return false;
#endif
        case Device::TPU:
        default:
            set(DeviceValidateCode::Unsupported); return false;
    }
}

}
