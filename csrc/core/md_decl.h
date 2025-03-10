//
// Created by aichao on 2025/2/26.
//

#pragma once

#if defined(_WIN32)
#ifdef MD_CXX_EXPORT
#define MODELDEPLOY_CXX_EXPORT __declspec(dllexport)
#else
#define MODELDEPLOY_CXX_EXPORT __declspec(dllimport)
#endif  // MD_CXX_EXPORT
#else
#define MODELDEPLOY_CXX_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32