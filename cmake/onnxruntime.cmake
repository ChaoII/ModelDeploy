#
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

set(onnxruntime_win_x64_static_1.20.1_mt_FILE_NAME "onnxruntime_win_x64_static_1.20.1_mt.zip")
set(onnxruntime_win_x64_static_1.20.1_md_FILE_NAME "onnxruntime_win_x64_static_1.20.1_md.zip")
set(onnxruntime_linux_x64_static_1.22.0_FILE_NAME "onnxruntime_linux_x64_static_1.22.0.zip")
set(onnxruntime_linux_aarch64_static_1.22.0_FILE_NAME "onnxruntime_linux_aarch64_static_1.22.0.zip")
set(ONNXRUNTIME_BASE_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master")
include(FetchContent)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    if (WITH_STATIC_CRT)
        set(ONNXRUNTIME_FILE_NAME ${onnxruntime_win_x64_static_1.20.1_mt_FILE_NAME})
        set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
        set(ONNXRUNTIME_HASH "SHA256=06c19e1c9214f9f7b7c780d14424696f1eee3c39724f738607caa33760fa4cb1")
    else ()
        set(ONNXRUNTIME_FILE_NAME ${onnxruntime_win_x64_static_1.20.1_md_FILE_NAME})
        set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
        set(ONNXRUNTIME_HASH "SHA256=8f1df4b53222ed4b92cd809157b219441ea52a9d74442a1254a9ff2e22852f78")
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_x64_static_1.22.0_FILE_NAME})
        set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
        set(ONNXRUNTIME_HASH "SHA256=8b39de5745c45baea301d47223164296bf5a3c14b9599afb92d829106d92aa7c")
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(ONNXRUNTIME_FILE_NAME ${onnxruntime_linux_aarch64_static_1.22.0_FILE_NAME})
        set(ONNXRUNTIME_URL ${ONNXRUNTIME_BASE_URL}/${ONNXRUNTIME_FILE_NAME})
        set(ONNXRUNTIME_HASH "SHA256=8b4e4da3183ba51ce4ac7558d9f6631cb3df7ffad9eb6ff048166167a8d79523")
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    message(FATAL_ERROR "Unsupported system :" ${CMAKE_SYSTEM_NAME})
endif ()

set(possible_file_locations
        $ENV{HOME}/Downloads/${ONNXRUNTIME_FILE_NAME}
        ${CMAKE_SOURCE_DIR}/${ONNXRUNTIME_FILE_NAME}
        ${CMAKE_BINARY_DIR}/${ONNXRUNTIME_FILE_NAME}
        /tmp/${ONNXRUNTIME_FILE_NAME})


foreach (f IN LISTS possible_file_locations)
    if (EXISTS ${f})
        set(ONNXRUNTIME_URL "${f}")
        file(TO_CMAKE_PATH "${ONNXRUNTIME_URL}" ONNXRUNTIME_URL)
        message(STATUS "Found local downloaded onnxruntime: ${ONNXRUNTIME_URL}")
        break()
    endif ()
endforeach ()

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
if (NOT onnxruntime_SOURCE_DIR)
    message(FATAL_ERROR "onnxruntime_SOURCE_DIR is not set after population")
endif ()

# for debug
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#    set(onnxruntime_SOURCE_DIR "E:/develop/onnxruntime-win-x64-1.22.0")
    set(onnxruntime_SOURCE_DIR "C:/Users/aichao/Downloads/onnxruntime-win-x64-gpu-1.20.1")
endif ()
include_directories(${onnxruntime_SOURCE_DIR}/include)
link_directories(${onnxruntime_SOURCE_DIR}/lib)
list(APPEND DEPENDS onnxruntime)