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
        set(ONNXRUNTIME_HASH "SHA256=06c260526302a5c896a9438b862c2374d8af8fe25e3be9d6205ea1234f145a48")
    else ()
        set(ONNXRUNTIME_FILE_NAME ${onnxruntime_win_x64_static_1_29_0_FILE_NAME})
        set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
        set(ONNXRUNTIME_HASH "SHA256=d23d9975bcce2a3e887597229d21e57e8c4ac9fa1a331deb5c6da2c21c9d7abd")
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (WITH_GPU)
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_x64_gpu_1_29_0_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=0a65719e8e8d2f5e12ab72b43282196734f61655deda9c9cb8b9666e0ac337fe")
        else ()
            message(FATAL_ERROR "Unsupported system arch : ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR} for GPU. Please set -DWITH_GPU=OFF")
        endif ()
    else ()
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_x64_static_1_29_0_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=f7a3955d3030dd897e6f420a8a1a1d14d14e347c66875279d14ec0c7cd7d56c5")
        elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
            set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_aarch64_static_1_29_0_FILE_NAME})
            set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
            set(ONNXRUNTIME_HASH "SHA256=abb5c3805c7e8bd580da443b928ceba4b2ccb48c9a1f91eeee862763c3f3adb7")
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





