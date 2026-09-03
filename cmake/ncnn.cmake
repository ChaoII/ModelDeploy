# ncnn 后端依赖（CPU + Vulkan）。
#
# 注意：当前仅 Windows x64 提供已确认可用的自动下载工件（modelscope），
# 其余平台 / 未确认工件一律 fail-fast——请设置 -DNCNN_ROOT 指向本地已解包目录。
# 本地解包：下载 ncnn 预编译包后解压，路径内需含 <arch>/lib/cmake（find_package 用）
# 与 <arch>/include（头文件用）。
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

set(NCNN_BASE_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master")
set(NCNN_LIBS ncnn)

# 平台 → 包内 arch 子目录
if (WIN32)
    set(NCNN_ARCH "x64")
elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(NCNN_ARCH "arm64")
else()
    set(NCNN_ARCH "x64")
endif()

if (NCNN_ROOT)  # 本地已解包目录，调试用
    message(STATUS "Using local NCNN_ROOT=${NCNN_ROOT}")
elseif (WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)  # 仅 Windows x64 有已确认下载工件
    include(FetchContent)
    set(NCNN_FILE_NAME "ncnn_win_x64_static_20260526.zip")
    FetchContent_Declare(ncnn
        URL ${NCNN_BASE_URL}/${NCNN_FILE_NAME}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_GetProperties(ncnn)
    if (NOT ncnn_POPULATED)
        message(STATUS "Downloading ncnn from ${NCNN_BASE_URL}/${NCNN_FILE_NAME}")
        FetchContent_Populate(ncnn)
    endif()
    set(NCNN_ROOT "${ncnn_SOURCE_DIR}")
else()
    message(FATAL_ERROR
        "ncnn 依赖在非 Windows-x64 平台（或未确认工件）不自动下载。"
        "请先本地解包 ncnn 预编译包，然后用 -DNCNN_ROOT=<解包路径> 指定 "
        "（解包路径内含 <arch>/lib/cmake 与 <arch>/include）。"
        "当前仅 Windows x64 有已确认的自动下载工件可用。")
endif()

set(NCNN_DIR "${NCNN_ROOT}/${NCNN_ARCH}")
find_package(ncnn CONFIG REQUIRED PATHS "${NCNN_DIR}/lib/cmake" NO_DEFAULT_PATH)
find_package(glslang CONFIG REQUIRED PATHS "${NCNN_DIR}/lib/cmake" NO_DEFAULT_PATH)
include_directories(${NCNN_DIR}/include)
