#pragma once
#include "core/md_decl.h"
#include "core/enum_variables.h"

namespace modeldeploy {
enum class DeviceValidateCode { Ok, Invalid, Unsupported };
MODELDEPLOY_CXX_EXPORT bool validate_pointer_device(const void* ptr, Device dev, int device_id,
                                                    DeviceValidateCode* out_code = nullptr);
}
