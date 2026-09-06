# MNN 后端依赖（单库静态包，3.6.1），与 onnxruntime 同范式：
# 配置时按平台差异从魔塔 Modelscope 自动下载，无需本地 *_ROOT / 本地 lib。
#
# 已上传工件（modelscope repo: ChaoII0987/ModelDeploy_cmake_deps）：
#   - Windows x64  : mnn_win_x64_static_3_6_1.zip
#   - Linux x64    : mnn_linux_x64_static_3_6_1.zip
#   - Linux aarch64: mnn_aarch64_static_3_6_1.zip
# 单库静态包（MNN_SEP_BUILD=OFF），即仅 MNN 一个导入目标，
# Express/OpenCL/Vulkan 后端已内联其中；无独立 CUDA 库（MNN CUDA 不随包分发）。
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

set(MNN_BASE_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master")
set(MNN_LIBS MNN)

include(FetchContent)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(MNN_FILE_NAME "mnn_win_x64_static_3_6_1.zip")
    set(MNN_HASH "SHA256=8b4661010939dfedba8bcbb0a5edd02744fb5ff5a03f22a75f16a85a33cb14a1")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /WHOLEARCHIVE:MNN")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(MNN_FILE_NAME "mnn_linux_x64_static_3_6_1.zip")
        set(MNN_HASH "SHA256=49562e7bc67c2e7011d224c50bc7a4e99a2a9c0a4ada57e639ffc1a77d323fc1")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(MNN_FILE_NAME "mnn_aarch64_static_3_6_1.zip")
        set(MNN_HASH "SHA256=d1637d66c783954e37dfb2717820f5cce5118b2edbde546cac7f8f1f1cdb59b4")
    else ()
        message(FATAL_ERROR "Unsupported system arch: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR} for MNN")
    endif ()
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--whole-archive -lMNN -Wl,--no-whole-archive")
else ()
    message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR} for MNN")
endif ()

set(MNN_URL "${MNN_BASE_URL}/${MNN_FILE_NAME}")

FetchContent_Declare(mnn
        URL ${MNN_URL}
        URL_HASH ${MNN_HASH}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(mnn)
if (NOT mnn_POPULATED)
    message(STATUS "Downloading MNN from ${MNN_URL}")
    FetchContent_Populate(mnn)
else ()
    message(STATUS "MNN is already populated")
endif ()
message(STATUS "MNN is downloaded to ${mnn_SOURCE_DIR}")

set(MNN_INC_DIR "${mnn_SOURCE_DIR}/include")
set(MNN_LIB_DIR "${mnn_SOURCE_DIR}/lib")
include_directories(${MNN_INC_DIR})
link_directories(${MNN_LIB_DIR})

find_library(MNN_LIB MNN
        PATHS "${MNN_LIB_DIR}"
        NO_DEFAULT_PATH
)

add_library(MNN STATIC IMPORTED GLOBAL)
set_target_properties(MNN PROPERTIES
        IMPORTED_LOCATION "${MNN_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MNN_INC_DIR}"
)

# 拷贝共享库（动态包需要；静态包无 *.dll/*.so 时为空操作）
file(GLOB MNN_SHARED_LIBS
        "${MNN_LIB_DIR}/*.dll"
        "${MNN_LIB_DIR}/*.so"
        "${MNN_LIB_DIR}/*.dylib"
)
file(COPY ${MNN_SHARED_LIBS} DESTINATION ${CMAKE_BINARY_DIR}/bin)
