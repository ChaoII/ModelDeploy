# GStreamer 查找模块：产出 GSTREAMER_INCLUDE_DIR 与 GSTREAMER_LIBS。
# Linux 优先走 pkg-config；Windows MSVC 无 pkg-config（或未找到），退回
# GSTREAMER_ROOT + find_library。找不到时 ENABLE_GSTREAMER 置 OFF（上层据此跳过）。

set(GSTREAMER_ROOT "C:/software/gstreamer/1.0/msvc_x86_64" CACHE PATH "GStreamer MSVC install root")
set(GSTREAMER_INCLUDE_DIR "")
set(GSTREAMER_LIBS "")

find_package(PkgConfig QUIET)
# Windows/MSVC 一般无可用 pkg-config（即便装了 MSYS 也常给出 -l 裸名，无法用于链接），
# 因此仅非 Windows 走 pkg-config，Windows 一律退回 GSTREAMER_ROOT + find_library。
if (PkgConfig_FOUND AND NOT WIN32)
    pkg_check_modules(GSTREAMER_APP QUIET gstreamer-app-1.0 gstreamer-1.0 gstreamer-video-1.0)
    if (GSTREAMER_APP_FOUND)
        set(GSTREAMER_INCLUDE_DIR ${GSTREAMER_APP_INCLUDE_DIRS})
        set(GSTREAMER_LIBS ${GSTREAMER_APP_LIBRARIES})
    endif ()
endif ()

# Windows/MSVC（或 pkg-config 不可用）：直接找头与 .lib。
# GStreamer 头跨三个根（gst/gst.h→gstreamer-1.0，glib.h→glib-2.0，glibconfig.h→lib/glib-2.0/include），
# 因此 GSTREAMER_INCLUDE_DIR 必须是目录列表（find_path 只给一个，故手动拼）。
if (NOT GSTREAMER_INCLUDE_DIR AND EXISTS "${GSTREAMER_ROOT}/include/gstreamer-1.0/gst/gst.h")
    list(APPEND GSTREAMER_INCLUDE_DIR "${GSTREAMER_ROOT}/include/gstreamer-1.0")
    list(APPEND GSTREAMER_INCLUDE_DIR "${GSTREAMER_ROOT}/include/glib-2.0")
    if (EXISTS "${GSTREAMER_ROOT}/lib/glib-2.0/include/glibconfig.h")
        list(APPEND GSTREAMER_INCLUDE_DIR "${GSTREAMER_ROOT}/lib/glib-2.0/include")
    endif ()
    foreach (lib gstreamer-1.0 gstapp-1.0 gstvideo-1.0 gstbase-1.0 gobject-2.0 glib-2.0)
        find_library(GSTREAMER_${lib}_LIB NAMES ${lib} ${lib}.lib
                PATHS ${GSTREAMER_ROOT}/lib NO_DEFAULT_PATH)
        if (GSTREAMER_${lib}_LIB)
            list(APPEND GSTREAMER_LIBS ${GSTREAMER_${lib}_LIB})
        endif ()
    endforeach ()
endif ()

# GStreamer CUDA（gstnvcodec / gstcuda）能力探测：GSTREAMER_HAS_CUDA=ON 时，
# gst/cuda 头链会 include <cuda.h>/<cudaD3D11.h>（需要 CUDA 工具包 include 目录与 Windows SDK），
# 且需链接 gstcuda-1.0。
set(GSTREAMER_HAS_CUDA OFF)
if (EXISTS "${GSTREAMER_ROOT}/include/gstreamer-1.0/gst/cuda/gstcudamemory.h"
    AND EXISTS "${GSTREAMER_ROOT}/include/gstreamer-1.0/gst/video/video.h")
    find_library(GSTREAMER_GSTCUDA_LIB NAMES gstcuda-1.0 gstcuda-1.0.lib
            PATHS ${GSTREAMER_ROOT}/lib NO_DEFAULT_PATH)
    # CUDA include：优先复用 WITH_GPU 已探测到的 CUDAToolkit 目录，否则退回常见默认路径。
    if (CUDAToolkit_INCLUDE_DIRS)
        set(GSTREAMER_CUDA_INCLUDE ${CUDAToolkit_INCLUDE_DIRS})
    else ()
        find_path(GSTREAMER_CUDA_INCLUDE cuda.h
                PATHS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
                NO_DEFAULT_PATH)
    endif ()
    if (GSTREAMER_GSTCUDA_LIB AND GSTREAMER_CUDA_INCLUDE)
        set(GSTREAMER_HAS_CUDA ON)
        list(APPEND GSTREAMER_INCLUDE_DIR ${GSTREAMER_CUDA_INCLUDE})
        list(APPEND GSTREAMER_LIBS ${GSTREAMER_GSTCUDA_LIB})
        message(STATUS "GStreamer CUDA (gstcuda) enabled: ${GSTREAMER_CUDA_INCLUDE}")
    else ()
        message(STATUS "GStreamer CUDA (gstcuda) unavailable (gstCUDA lib/include not found)")
    endif ()
endif ()

# Jetson L4T NvBufSurface（nvv4l2decoder 零拷贝设备直通）能力探测：GSTREAMER_HAS_NVBUF=ON 时
# gst_decoder 编译 L4T 分支（引入 /usr/src/jetson_multimedia_api/include 与 libnvbufsurface）。
# 仅 Linux/L4T（Jetson）可能命中；桌面/Windows 不会误启。
set(GSTREAMER_HAS_NVBUF OFF)
if (NOT WIN32 AND GSTREAMER_INCLUDE_DIR)
    find_path(NVBUF_INCLUDE_DIR nvbufsurface.h
            PATHS "/usr/src/jetson_multimedia_api/include" NO_DEFAULT_PATH)
    find_library(NVBUF_LIBRARY NAMES nvbufsurface
            PATHS "/usr/lib/aarch64-linux-gnu/tegra" "/usr/lib/aarch64-linux-gnu"
                  "/usr/local/lib" "/usr/lib" NO_DEFAULT_PATH)
    if (NVBUF_INCLUDE_DIR AND NVBUF_LIBRARY)
        set(GSTREAMER_HAS_NVBUF ON)
        list(APPEND GSTREAMER_INCLUDE_DIR ${NVBUF_INCLUDE_DIR})
        list(APPEND GSTREAMER_LIBS ${NVBUF_LIBRARY})
        message(STATUS "GStreamer L4T NvBufSurface (nvbufsurface) enabled: ${NVBUF_INCLUDE_DIR}")
    else ()
        message(STATUS "GStreamer L4T NvBufSurface unavailable (nvbufsurface.h/lib not found)")
    endif ()
endif ()

if (NOT GSTREAMER_INCLUDE_DIR OR NOT GSTREAMER_LIBS)
    message(WARNING "GStreamer not found at ${GSTREAMER_ROOT}; ENABLE_GSTREAMER disabled")
    set(ENABLE_GSTREAMER OFF)
endif ()
