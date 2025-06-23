#
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

set(onnxruntime_win_x64_static_1_20_1_mt_FILE_NAME "onnxruntime_win_x64_static_1_20_1_mt.zip")
set(onnxruntime_win_x64_static_1_20_1_md_FILE_NAME "onnxruntime_win_x64_static_1_20_1_md.zip")
set(onnxruntime_win_x64_gpu_1_22_0_FILE_NAME "onnxruntime-win-x64-gpu-1_22_0.zip")
set(onnxruntime_linux_x64_static_1_22_0_FILE_NAME "onnxruntime_linux_x64_static_1_22_0.zip")
set(onnxruntime_linux_aarch64_static_1_22_0_FILE_NAME "onnxruntime_linux_aarch64_static_1_22_0.zip")
set(onnxruntime_linux_x64_gpu_1_22_0_FILE_NAME "onnxruntime-linux-x64-gpu-1_22_0.zip")


set(ONNXRUNTIME_BASE_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master")
include(FetchContent)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    if (WITH_GPU)
        set(ONNXRUNTIME_FILE_NAME ${onnxruntime_win_x64_gpu_1_22_0_FILE_NAME})
        set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
        set(ONNXRUNTIME_HASH "SHA256=1ba28359814b21108fc66e8d9154e95c88c3d92b3a4334f408d1d51d79b038b7")
    else ()
        if (WITH_STATIC_CRT)
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_win_x64_static_1_20_1_mt_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=c6c79a9e73170bc5129492a71f60b9ffce3b17231dc198cbb5ab10daa5d60582")
        else ()
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_win_x64_static_1_20_1_md_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=ad8f8f741a1793bb78f81df72541e81de67e26e70168e7ca9e07b246b1a180de")
        endif ()
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (WITH_GPU)
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_x64_gpu_1_22_0_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=e891c98e424bb5064034c312a2ed659e37f49bccc6f25b9c86d7fc91104d5d19")
        else ()
            message(FATAL_ERROR "Unsupported system arch : ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR} for GPU. Please set -DWITH_GPU=OFF")
        endif ()
    else ()
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_x64_static_1_22_0_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=970f86c7760ea259eecbe76bff8cf9cfb90d7379c164562fd506f7e8611ebed9")
        elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_aarch64_static_1_22_0_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=4252cb8f804236b0abc0b6d2a29421d2091e6aa37bd3af9d694c6acc66693c25")
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

# 拷贝到 ${CMAKE_BINARY_DIR}/bin 或你指定的 bin 目录
file(GLOB ORT_SHARED_LIBS
        "${onnxruntime_SOURCE_DIR}/lib/*.dll"
        "${onnxruntime_SOURCE_DIR}/lib/*.so"
        "${onnxruntime_SOURCE_DIR}/lib/*.dylib"
)
file(COPY ${ORT_SHARED_LIBS} DESTINATION ${CMAKE_BINARY_DIR}/bin)





