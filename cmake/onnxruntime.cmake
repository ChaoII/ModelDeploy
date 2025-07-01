#
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

set(onnxruntime_win_x64_static_1_20_1_mt_FILE_NAME "onnxruntime_win_x64_static_1_20_1_mt.zip")
set(onnxruntime_win_x64_static_1_20_1_md_FILE_NAME "onnxruntime_win_x64_static_1_20_1_md.zip")
set(onnxruntime_win_x64_gpu_1_22_0_FILE_NAME "onnxruntime_win_x64_gpu_1_22_0.zip")
set(onnxruntime_linux_x64_static_1_22_0_FILE_NAME "onnxruntime_linux_x64_static_1_22_0.zip")
set(onnxruntime_linux_aarch64_static_1_22_0_FILE_NAME "onnxruntime_linux_aarch64_static_1_22_0.zip")
set(onnxruntime_linux_x64_gpu_1_22_0_FILE_NAME "onnxruntime_linux_x64_gpu_1_22_0.zip")


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
            set(ONNXRUNTIME_HASH "SHA256=750854123289251900beb397dd7938d060dc4e945a1b074070d152e88ec09af5")
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
            set(ONNXRUNTIME_HASH "SHA256=8c2fe08a3e3b0cee84503e4b76c30787b21ba206567db27cc1d63ab0026c0d56")
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





