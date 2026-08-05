# 查找 SOPHON-Sail（算能 TPU 推理 SDK），找不到时禁用 SOPHGO 后端。
# 用法：-DSOPHGO_SDK_DIR=/opt/sophon/sophon-sail 或环境变量 SOPHGO_SDK_DIR
# 通常 sail 安装路径：<SDK>/include/sail/engine.h + <SDK>/lib/libsail.so

if (NOT DEFINED SOPHGO_SDK_DIR)
    set(SOPHGO_SDK_DIR "$ENV{SOPHGO_SDK_DIR}")
endif ()

# 优先 libsophon-current（与设备驱动匹配的版本），其次官方 0.5.3（需配套 0.5.3 驱动）
set(SOPHON_LIBSOPHON_SEARCH_HINTS "/opt/sophon/libsophon-current" "/opt/sophon/libsophon-0.5.3")
set(SOPHON_LIBSOPHON_LIB_SEARCH_HINTS "")
foreach (_h ${SOPHON_LIBSOPHON_SEARCH_HINTS})
    if (EXISTS "${_h}/lib")
        list(APPEND SOPHON_LIBSOPHON_LIB_SEARCH_HINTS "${_h}/lib")
    endif ()
endforeach ()

set(SOPHGO_FOUND OFF)

if (SOPHGO_SDK_DIR)
    find_path(SOPHON_SAIL_INCLUDE_DIR sail/engine.h
        HINTS "${SOPHGO_SDK_DIR}"
        PATH_SUFFIXES include sail/include ../include ../sail/include)
    find_library(SOPHON_SAIL_LIB NAMES sail
        HINTS "${SOPHGO_SDK_DIR}"
        PATH_SUFFIXES lib sail/lib ../lib ../sail/lib)
    if (SOPHON_SAIL_INCLUDE_DIR AND SOPHON_SAIL_LIB)
        set(SOPHGO_FOUND ON)
        add_definitions(-DENABLE_SOPHGO)
        include_directories(${SOPHON_SAIL_INCLUDE_DIR})
        # sail 头文件依赖 libsophon（bmlib_runtime.h / bmruntime_interface.h）
        find_path(SOPHON_LIBSOPHON_INCLUDE_DIR bmlib_runtime.h
            HINTS "${SOPHON_LIBSOPHON_SEARCH_HINTS}" "${SOPHGO_SDK_DIR}" "/opt/sophon"
            PATH_SUFFIXES include libsophon-current/include ../include)
        if (SOPHON_LIBSOPHON_INCLUDE_DIR)
            include_directories(${SOPHON_LIBSOPHON_INCLUDE_DIR})
            message(STATUS "libsophon include: ${SOPHON_LIBSOPHON_INCLUDE_DIR}")
        endif ()
        # sail 依赖 sophon bmrt/libbmcv/libbmlib，链接时一并带上（从 libsophon 目录查找）
        find_library(SOPHON_BMRT_LIB NAMES bmrt HINTS ${SOPHON_LIBSOPHON_LIB_SEARCH_HINTS} "${SOPHGO_SDK_DIR}" PATH_SUFFIXES lib ../lib)
        find_library(SOPHON_BMCV_LIB NAMES bmcv HINTS ${SOPHON_LIBSOPHON_LIB_SEARCH_HINTS} "${SOPHGO_SDK_DIR}" PATH_SUFFIXES lib ../lib)
        find_library(SOPHON_BMLIB_LIB NAMES bmlib HINTS ${SOPHON_LIBSOPHON_LIB_SEARCH_HINTS} "${SOPHGO_SDK_DIR}" PATH_SUFFIXES lib ../lib)
        set(SOPHGO_LIBS ${SOPHON_SAIL_LIB})
        if (SOPHON_BMRT_LIB)
            list(APPEND SOPHGO_LIBS ${SOPHON_BMRT_LIB})
        endif ()
        if (SOPHON_BMCV_LIB)
            list(APPEND SOPHGO_LIBS ${SOPHON_BMCV_LIB})
        endif ()
        if (SOPHON_BMLIB_LIB)
            list(APPEND SOPHGO_LIBS ${SOPHON_BMLIB_LIB})
        endif ()
        message(STATUS "SOPHON-Sail found at ${SOPHON_SAIL_INCLUDE_DIR}, SOPHGO backend enabled")
    else ()
        message(STATUS "SOPHON-Sail headers/libs not found under ${SOPHGO_SDK_DIR}, SOPHGO backend disabled")
    endif ()
else ()
    message(STATUS "SOPHGO_SDK_DIR not set, SOPHGO backend disabled (set -DSOPHGO_SDK_DIR=/path/to/sophon-sail)")
endif ()
