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

if (NOT GSTREAMER_INCLUDE_DIR OR NOT GSTREAMER_LIBS)
    message(WARNING "GStreamer not found at ${GSTREAMER_ROOT}; ENABLE_GSTREAMER disabled")
    set(ENABLE_GSTREAMER OFF)
endif ()
