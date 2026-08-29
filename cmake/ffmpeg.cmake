# FFmpeg 查找模块：产出 FFMPEG_INCLUDE_DIR 与 FFMPEG_LIBS（模块化 application/CMakeLists.txt 的 find 逻辑）。
# 用法：option(BUILD_VIDEO ...) 为 ON 时 include 本文件。
# 找不到 FFmpeg 时把 BUILD_VIDEO 置 OFF（优雅禁用），由上层据此跳过视频编译。

set(FFMPEG_ROOT "E:/develop/ffmpeg" CACHE PATH "FFmpeg installation root")
find_path(FFMPEG_INCLUDE_DIR libavcodec/avcodec.h PATHS ${FFMPEG_ROOT}/include NO_DEFAULT_PATH)

set(FFMPEG_LIBS "")
foreach (lib avcodec avformat avutil swscale avdevice)
    find_library(FFMPEG_${lib}_LIBRARY NAMES ${lib} ${lib}.lib
            PATHS ${FFMPEG_ROOT}/lib NO_DEFAULT_PATH)
    if (FFMPEG_${lib}_LIBRARY)
        list(APPEND FFMPEG_LIBS ${FFMPEG_${lib}_LIBRARY})
    endif ()
endforeach ()

if (NOT FFMPEG_INCLUDE_DIR OR NOT FFMPEG_LIBS)
    # 非 Windows（Linux/Jetson/Sophgo）回退到 pkg-config 系统 FFmpeg
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
        pkg_check_modules(FFMPEG QUIET libavcodec libavformat libavutil libswscale)
        if (FFMPEG_FOUND)
            set(FFMPEG_INCLUDE_DIR "${FFMPEG_INCLUDE_DIRS}")
            set(FFMPEG_LIBS "${FFMPEG_LIBRARIES}")
        endif ()
    endif ()
endif ()

# Sophgo（算能）硬件编解码：优先链 sophon-ffmpeg 的运行库（内含 h264_bm/h265_bm/hevc_bm 等
# BM 硬件编码/解码 wrapper）。该包不带头文件，故头文件仍用系统/上述查找结果，仅覆盖链接库。
# 注意：与 app 侧 FFmpeg 必须链同一套，避免一个进程内双 libavcodec 符号冲突。
if (FFMPEG_LIBS)
    foreach (_sd /opt/sophon/sophon-ffmpeg-latest /opt/sophon/sophon-ffmpeg_2.2.0)
        if (EXISTS "${_sd}/lib/libavcodec.so")
            set(SOPHON_FFMPEG_LIB "${_sd}/lib")
            break()
        endif ()
    endforeach ()
endif ()
if (SOPHON_FFMPEG_LIB)
    set(FFMPEG_LIBS "")
    foreach (_l avcodec avformat avutil swscale avdevice)
        find_library(FFMPEG_SOPHON_${_l} NAMES ${_l} PATHS ${SOPHON_FFMPEG_LIB} NO_DEFAULT_PATH)
        if (FFMPEG_SOPHON_${_l})
            list(APPEND FFMPEG_LIBS ${FFMPEG_SOPHON_${_l}})
        endif ()
    endforeach ()
    message(STATUS "FFmpeg: linking sophon-ffmpeg hardware libs under ${SOPHON_FFMPEG_LIB}")
endif ()

if (NOT FFMPEG_INCLUDE_DIR OR NOT FFMPEG_LIBS)
    message(WARNING "FFmpeg not found (Windows: set FFMPEG_ROOT; Linux: install libav*-dev); BUILD_VIDEO disabled")
    set(BUILD_VIDEO OFF)
endif ()
