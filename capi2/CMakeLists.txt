# C API v2（纯 C99 头 + 统一分发）
if (NOT BUILD_CAPI)
    return()
endif ()

file(GLOB_RECURSE CAPI2_SOURCE ${CMAKE_SOURCE_DIR}/capi2/*.cpp)
list(APPEND CAPI_SOURCE ${CAPI2_SOURCE})
