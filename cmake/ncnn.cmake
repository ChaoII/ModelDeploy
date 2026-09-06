# ncnn 后端依赖（CPU + Vulkan，静态库），与 onnxruntime 同范式：
# 配置时按平台差异从魔塔 Modelscope 自动下载，无需本地 *_ROOT / 本地 lib。
#
# 已上传工件（modelscope repo: ChaoII0987/ModelDeploy_cmake_deps）：
#   - Windows x64  : ncnn_win_x64_sttaic_20260526.zip
#   - Linux x64    : ncnn_linux_x64_static_20260526.zip
#   - Linux aarch64: ncnn_aarch64_static_20260526.zip
#
# 解包后主目录即含 include 与 lib，lib 内含 cmake/ncnn 与 cmake/glslang；
# ncnn 目标自动携带对 glslang 的链接依赖。其余平台一律 fail-fast。
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

set(NCNN_BASE_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master")
set(NCNN_LIBS ncnn)

include(FetchContent)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(NCNN_FILE_NAME "ncnn_win_x64_sttaic_20260526.zip")
    set(NCNN_HASH "SHA256=88d58700b067e1b30fb7e3bfa0792cae198c149dfdcdff71be282c5d2d956d3e")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(NCNN_FILE_NAME "ncnn_linux_x64_static_20260526.zip")
        set(NCNN_HASH "SHA256=a0095865b47c1be1d6fa70215ecf182179d49aca064d301637835b3e700e35a8")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(NCNN_FILE_NAME "ncnn_aarch64_static_20260526.zip")
        set(NCNN_HASH "SHA256=afd21e11672a5ffbc88027b62be5fc3cb90bc03adf299e60d49854434623509b")
    else ()
        message(FATAL_ERROR "Unsupported system arch: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR} for ncnn")
    endif ()
else ()
    message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR} for ncnn")
endif ()

set(NCNN_URL "${NCNN_BASE_URL}/${NCNN_FILE_NAME}")

FetchContent_Declare(ncnn
        URL ${NCNN_URL}
        URL_HASH ${NCNN_HASH}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_GetProperties(ncnn)
if (NOT ncnn_POPULATED)
    message(STATUS "Downloading ncnn from ${NCNN_URL}")
    FetchContent_Populate(ncnn)
endif ()

# ncnn 与 glslang 的 CMake config 同居一目录；ncnn 目标自动携带对 glslang 的链接依赖
find_package(ncnn CONFIG REQUIRED PATHS "${ncnn_SOURCE_DIR}/lib/cmake" NO_DEFAULT_PATH)
find_package(glslang CONFIG REQUIRED PATHS "${ncnn_SOURCE_DIR}/lib/cmake" NO_DEFAULT_PATH)
include_directories(${ncnn_SOURCE_DIR}/include)
