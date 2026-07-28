#pragma once

#include "capi/common/md_decl.h"
#include "capi/common/md_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Clone a model. Creates an independent copy with its own runtime.
/// @param model [out] The cloned model (must not be initialized)
/// @param from [in] The source model to clone from
/// @return Status code
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_clone_model(MDModel* model, const MDModel* from);

#ifdef __cplusplus
}
#endif
