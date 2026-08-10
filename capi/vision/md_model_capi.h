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

/// Query whether a model has been successfully initialized.
/// @param model [in] The model handle
/// @return 1 if initialized, 0 otherwise
MODELDEPLOY_CAPI_EXPORT int md_model_is_initialized(const MDModel* model);

/// Get the model's name string (e.g. "UltralyticsDet"). Caller frees with free().
/// @param model [in] The model handle
/// @return malloc'd name string, or nullptr if model invalid
MODELDEPLOY_CAPI_EXPORT char* md_model_name(const MDModel* model);

#ifdef __cplusplus
}
#endif
