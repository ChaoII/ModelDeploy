//
// Created by aichao on 2025/2/26.
//

#pragma once

#if defined(_WIN32)
#ifdef MD_CAPI
#define MODELDEPLOY_CAPI_EXPORT __declspec(dllexport)
#else
#define MODELDEPLOY_CAPI_EXPORT __declspec(dllimport)
#endif  // MD_CAPI
#else
#define MODELDEPLOY_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32