# 查找 SOPHON libsophon（算能 TPU 推理 SDK：bmlib/bmrt/bmcv），找不到时禁用 SOPHGO 后端。
# SophgoBackend 用 bmrt 直接推理（不依赖 SOPHON-Sail），仅需 libsophon：
#   - bmlib_runtime.h / bmruntime_interface.h / bmdef.h（头文件）
#   - libbmlib / libbmrt / libbmcv（库）
# 用法：-DSOPHGO_SDK_DIR=/opt/sophon/libsophon-current 或环境变量 SOPHGO_SDK_DIR
# （也可省略，自动探测 /opt/sophon/libsophon-*）

if (NOT DEFINED SOPHGO_SDK_DIR)
    set(SOPHGO_SDK_DIR "$ENV{SOPHGO_SDK_DIR}")
endif ()

# 优先 libsophon-current（与设备驱动匹配的版本），其次官方 0.5.3（需配套 0.5.3 驱动）
set(SOPHON_LIBSOPHON_SEARCH_HINTS "/opt/sophon/libsophon-current" "/opt/sophon/libsophon-0.5.3")
if (SOPHGO_SDK_DIR)
    list(INSERT SOPHON_LIBSOPHON_SEARCH_HINTS 0 "${SOPHGO_SDK_DIR}")
endif ()
set(SOPHON_LIBSOPHON_LIB_SEARCH_HINTS "")
foreach (_h ${SOPHON_LIBSOPHON_SEARCH_HINTS})
    if (EXISTS "${_h}/lib")
        list(APPEND SOPHON_LIBSOPHON_LIB_SEARCH_HINTS "${_h}/lib")
    endif ()
endforeach ()

set(SOPHGO_FOUND OFF)

find_path(SOPHON_LIBSOPHON_INCLUDE_DIR bmlib_runtime.h
    HINTS ${SOPHON_LIBSOPHON_SEARCH_HINTS}
    PATH_SUFFIXES include libsophon-current/include ../include)
find_library(SOPHON_BMRT_LIB NAMES bmrt HINTS ${SOPHON_LIBSOPHON_LIB_SEARCH_HINTS} PATH_SUFFIXES lib ../lib)
find_library(SOPHON_BMCV_LIB NAMES bmcv HINTS ${SOPHON_LIBSOPHON_LIB_SEARCH_HINTS} PATH_SUFFIXES lib ../lib)
find_library(SOPHON_BMLIB_LIB NAMES bmlib HINTS ${SOPHON_LIBSOPHON_LIB_SEARCH_HINTS} PATH_SUFFIXES lib ../lib)

if (SOPHON_LIBSOPHON_INCLUDE_DIR AND SOPHON_BMRT_LIB AND SOPHON_BMLIB_LIB AND SOPHON_BMCV_LIB)
    set(SOPHGO_FOUND ON)
    add_definitions(-DENABLE_SOPHGO)
    include_directories(${SOPHON_LIBSOPHON_INCLUDE_DIR})
    set(SOPHGO_LIBS ${SOPHON_BMRT_LIB} ${SOPHON_BMCV_LIB} ${SOPHON_BMLIB_LIB})
    message(STATUS "libsophon(bmrt) found: include=${SOPHON_LIBSOPHON_INCLUDE_DIR} libs=${SOPHON_LIBSOPHON_LIB_SEARCH_HINTS}")
    message(STATUS "SOPHGO backend enabled (bmrt/bmcv/bmlib direct, no sail)")
else ()
    message(STATUS "SOPHON libsophon (bmlib_runtime.h / libbmrt / libbmcv / libbmlib) not found under ${SOPHON_LIBSOPHON_SEARCH_HINTS}, SOPHGO backend disabled")
endif ()
