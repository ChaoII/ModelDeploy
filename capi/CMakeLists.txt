# C API v2（纯 C99 头 + 统一分发）
if (NOT BUILD_CAPI)
    return()
endif ()

file(GLOB_RECURSE CAPI_SOURCE ${CMAKE_SOURCE_DIR}/capi/*.cpp)
list(APPEND CAPI_SOURCE ${CAPI_SOURCE})
