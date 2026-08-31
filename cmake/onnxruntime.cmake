message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

set(onnxruntime_win_x64_static_1_29_0_FILE_NAME "onnxruntime_win_x64_static_1_29_0.zip")
set(onnxruntime_win_x64_gpu_1_29_0_FILE_NAME "onnxruntime_win_x64_gpu_1_29_0.zip")
set(onnxruntime_linux_x64_static_1_29_0_FILE_NAME "onnxruntime_linux_x64_static_1_29_0.zip")
set(onnxruntime_linux_x64_gpu_1_29_0_FILE_NAME "onnxruntime_linux_x64_gpu_1_29_0.zip")
set(onnxruntime_linux_aarch64_static_1_29_0_FILE_NAME "onnxruntime_linux_aarch64_static_1_29_0.zip")


set(ONNXRUNTIME_BASE_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master")
include(FetchContent)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    if (WITH_GPU)
        set(ONNXRUNTIME_FILE_NAME ${onnxruntime_win_x64_gpu_1_29_0_FILE_NAME})
        set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
        set(ONNXRUNTIME_HASH "SHA256=8a7d29232cff015e408caf24086ba5d4dcabdfdbd5d3fcdaa799b6de3142d3a6")
    else ()
        set(ONNXRUNTIME_FILE_NAME ${onnxruntime_win_x64_static_1_29_0_FILE_NAME})
        set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
        set(ONNXRUNTIME_HASH "SHA256=41494a48919519f51098ca501bcf0d80e74f66c000878d357d6a93c53df93f34")
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (WITH_GPU)
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_x64_gpu_1_29_0_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=091ef6724ef2c98012caa92dd9074aacb51b43ed6e18acf9cf545ff36c93735c")
        else ()
            message(FATAL_ERROR "Unsupported system arch : ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR} for GPU. Please set -DWITH_GPU=OFF")
        endif ()
    else ()
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_x64_static_1_29_0_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=601b6b65d5865fa2b8917f8065c892c1a1e8a3a27deacfd4e0f77a48e731d0cd")
        elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_aarch64_static_1_29_0_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=160d47d8b2e0b63cd052a612e31d623088193fee5f9faf5e5cd5a3137f5cf423")
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
)

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





