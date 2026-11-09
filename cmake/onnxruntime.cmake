message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

set(onnxruntime_win_x64_static_1_29_0_FILE_NAME "onnxruntime_win_x64_static_1_29_0.zip")
set(onnxruntime_win_x64_gpu_1_29_0_FILE_NAME "onnxruntime_win_x64_gpu_1_29_0.zip")
set(onnxruntime_win_x64_gpu_1_29_0_cuda13_FILE_NAME "onnxruntime_win_x64_gpu_1_29_0_cuda13.zip")
# Linux 侧改用 csukuangfj/onnxruntime-libs 的 glibc2.28 静态包（AlmaLinux 8 用户空间构建）：
# 自建包是在 Ubuntu 22.04/24.04 上编的，会引入 GLIBC_2.35+ 符号，glibc 2.28 目标机无法加载。
set(onnxruntime_linux_x64_static_1_29_1_glibc2_28_FILE_NAME "onnxruntime_linux_x64_static_1_29_1_glibc2_28.zip")
# GPU 包按 CUDA 大版本区分（csukuangfj/onnxruntime-libs 的 patched 构建）：
# cu12 包依赖 libcublas.so.12，cu13 包依赖 libcublas.so.13，必须与 CUDAToolkit 大版本匹配
set(onnxruntime_linux_x64_gpu_1_29_1_cuda12_FILE_NAME "onnxruntime_linux_x64_gpu_1_29_1_cuda12.zip")
set(onnxruntime_linux_x64_gpu_1_29_1_cuda13_FILE_NAME "onnxruntime_linux_x64_gpu_1_29_1_cuda13.zip")
set(onnxruntime_linux_aarch64_static_1_29_1_glibc2_28_FILE_NAME "onnxruntime_linux_aarch64_static_1_29_1_glibc2_28.zip")


set(ONNXRUNTIME_BASE_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master")
include(FetchContent)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    if (WITH_GPU)
        # 按 CUDAToolkit 大版本选 GPU 包：
        #   cu12 包 = 自建（cublas64_12），cu13 包 = Microsoft.ML.OnnxRuntime.Gpu.Windows 1.29.0
        #   （官方 1.27+ GPU 包默认 CUDA 13.0 构建，cublas64_13）
        if (CUDA_MAJOR_VERSION GREATER_EQUAL 13)
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_win_x64_gpu_1_29_0_cuda13_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=06ec04215bfa314fafd4694a4fb4948207657ca24baaaaec05ddd38578306949")
        else ()
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_win_x64_gpu_1_29_0_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=06c260526302a5c896a9438b862c2374d8af8fe25e3be9d6205ea1234f145a48")
        endif ()
    else ()
        set(ONNXRUNTIME_FILE_NAME ${onnxruntime_win_x64_static_1_29_0_FILE_NAME})
        set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
        set(ONNXRUNTIME_HASH "SHA256=d23d9975bcce2a3e887597229d21e57e8c4ac9fa1a331deb5c6da2c21c9d7abd")
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (WITH_GPU)
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
            # 按 CUDAToolkit 大版本选 GPU 包（CMakeLists 在 include 本文件前已 find_package(CUDAToolkit)）
            if (CUDA_MAJOR_VERSION GREATER_EQUAL 13)
                set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_x64_gpu_1_29_1_cuda13_FILE_NAME})
                set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
                set(ONNXRUNTIME_HASH "SHA256=fccae695cf6b74ff8d10d78c6fb88c1c2f92074ea77a8d95218faf9e52fb7cda")
            else ()
                set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_x64_gpu_1_29_1_cuda12_FILE_NAME})
                set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
                set(ONNXRUNTIME_HASH "SHA256=2faff427f1dbb0174726417ec2ffa39058182a0b653341ee373f8aaa9958f6f3")
            endif ()
        else ()
            message(FATAL_ERROR "Unsupported system arch : ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR} for GPU. Please set -DWITH_GPU=OFF")
        endif ()
    else ()
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_x64_static_1_29_1_glibc2_28_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=c773278a4d7451ce5a6bb9c842ede52f390da24c87c0facccd52d01bc32549d2")
        elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_aarch64_static_1_29_1_glibc2_28_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=c5be1b8964e4860231e44130184430753d5afd86b8a6877c133756bb84eb6419")
        else ()
            message(FATAL_ERROR "Unsupported system arch:" ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR})
        endif ()
    endif ()
else ()
    message(FATAL_ERROR "Unsupported system :" ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR})
endif ()

FetchContent_Declare(onnxruntime
        URL
        ${ONNXRUNTIME_URL}
        URL_HASH ${ONNXRUNTIME_HASH}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(onnxruntime)
if (NOT onnxruntime_POPULATED)
    message(STATUS "Downloading onnxruntime from ${ONNXRUNTIME_URL}")
    FetchContent_Populate(onnxruntime)
else ()
    message(STATUS "onnxruntime is already populated")
endif ()
message(STATUS "onnxruntime is downloaded to ${onnxruntime_SOURCE_DIR}")

include_directories(${onnxruntime_SOURCE_DIR}/include)
link_directories(${onnxruntime_SOURCE_DIR}/lib)

find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS "${onnxruntime_SOURCE_DIR}/lib"
        NO_DEFAULT_PATH
        NO_CMAKE_FIND_ROOT_PATH
)

if (NOT ONNXRUNTIME_LIB OR ONNXRUNTIME_LIB MATCHES "NOTFOUND")
    message(FATAL_ERROR "onnxruntime library not found under ${onnxruntime_SOURCE_DIR}/lib (download/extract failed?)")
endif ()

add_library(onnxruntime::onnxruntime STATIC IMPORTED GLOBAL)


set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
)

# 拷贝到 ${CMAKE_BINARY_DIR}/bin 或你指定的 bin 目录
if (WIN32)
    file(GLOB ORT_SHARED_LIBS "${onnxruntime_SOURCE_DIR}/lib/*.dll")
elseif (APPLE)
    file(GLOB ORT_SHARED_LIBS "${onnxruntime_SOURCE_DIR}/lib/*.dylib")
else ()
    file(GLOB ORT_SHARED_LIBS "${onnxruntime_SOURCE_DIR}/lib/*.so" "${onnxruntime_SOURCE_DIR}/lib/*.so.*")
endif ()
file(COPY ${ORT_SHARED_LIBS} DESTINATION ${CMAKE_BINARY_DIR}/bin)





