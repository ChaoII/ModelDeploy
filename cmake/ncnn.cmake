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
else()
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
endif()

set(NCNN_DIR "${NCNN_ROOT}/${NCNN_ARCH}")
find_package(ncnn CONFIG REQUIRED PATHS "${NCNN_DIR}/lib/cmake" NO_DEFAULT_PATH)
find_package(glslang CONFIG REQUIRED PATHS "${NCNN_DIR}/lib/cmake" NO_DEFAULT_PATH)
include_directories(${NCNN_DIR}/include)
