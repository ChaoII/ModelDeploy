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

if (NOT FFMPEG_INCLUDE_DIR OR NOT FFMPEG_LIBS)
    message(WARNING "FFmpeg not found (Windows: set FFMPEG_ROOT; Linux: install libav*-dev); BUILD_VIDEO disabled")
    set(BUILD_VIDEO OFF)
endif ()
