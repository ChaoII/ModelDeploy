#
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

# --- 联网下载预编译 OpenCV5 ---
set(opencv_5_x_win_x64_static_md_FILE_NAME "opencv_5_x_win_x64_static.zip")
set(opencv_5_x_linux_x64_static_FILE_NAME "opencv_5_x_linux_x64_static.zip")
set(opencv_5_x_linux_aarch64_static_FILE_NAME "opencv_5_x_linux_aarch64_static.zip")
set(OPENCV_BASE_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master")
include(FetchContent)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(OPENCV_FILE_NAME ${opencv_5_x_win_x64_static_md_FILE_NAME})
    set(OPENCV_URL ${OPENCV_BASE_URL}/${OPENCV_FILE_NAME})
    set(OPENCV_HASH "SHA256=73140ac93d84507e7bb5698160f85f962c51cd6043597389792fffc65760e0fd")
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(OPENCV_FILE_NAME ${opencv_5_x_linux_x64_static_FILE_NAME})
        set(OPENCV_URL ${OPENCV_BASE_URL}/${OPENCV_FILE_NAME})
        set(OPENCV_HASH "SHA256=e912077b5f64829ca3cb8a1b6edaa557878e823efb6def58eeaaebd0f2fc2a37")
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(OPENCV_FILE_NAME ${opencv_5_x_linux_aarch64_static_FILE_NAME})
        set(OPENCV_URL ${OPENCV_BASE_URL}/${OPENCV_FILE_NAME})
        set(OPENCV_HASH "SHA256=c46f4a97ddb0ccc6abfa9d54aec633bdb7f970ee38588ac1081c495d83f785a9")
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    message(FATAL_ERROR "Unsupported system :" ${CMAKE_SYSTEM_NAME})
endif ()

FetchContent_Declare(opencv
        URL
        ${OPENCV_URL}
        URL_HASH ${OPENCV_HASH}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(opencv)
if (NOT opencv_POPULATED)
    message(STATUS "Downloading opencv from ${OPENCV_URL}")
    FetchContent_Populate(opencv)
else ()
    message(STATUS "opencv is already populated")
endif ()
message(STATUS "opencv is downloaded to ${opencv_SOURCE_DIR}")
if (NOT opencv_SOURCE_DIR)
    message(FATAL_ERROR "opencv_SOURCE_DIR is not set after population")
endif ()

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(OpenCV_DIR "${opencv_SOURCE_DIR}/x64/vc17/staticlib")
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(OpenCV_DIR "${opencv_SOURCE_DIR}/lib/cmake/opencv5")
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(OpenCV_DIR "${opencv_SOURCE_DIR}/lib64/cmake/opencv5")
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    message(FATAL_ERROR "Unsupported system :" ${CMAKE_SYSTEM_NAME})
endif ()

if (NOT DEFINED OpenCV_DIR)
    find_package(OpenCV CONFIG REQUIRED)
else ()
    find_package(OpenCV CONFIG REQUIRED)
endif ()

if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "build BUILD_VISION depends on opencv 5.0 from https://www.modelscope.cn (downloaded to _deps/opencv-src), network download failed")
endif ()
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Sophgo 平台：OpenCV 捆绑的 libjpeg-turbo 的 jsimd(SIMD) 符号默认未被链接器拉入，
# 运行时被 libbmcv.so 导出的 jsimd 符号劫持，导致保存 .jpg 整体变暗。
# 用 --whole-archive 强制把捆绑 libjpeg-turbo 的 jsimd 静态编入 SDK，使其不再走全局符号解析。
# 通过 OpenCV imported target "libjpeg-turbo" 取静态库路径（$<TARGET_FILE> 在链接期求值），
# 不依赖硬编码 _deps 路径。该变量供顶层 CMakeLists.txt 在 target_link_libraries 后消费。
set(MD_OCV_JPEG_TURBO_WHOLE_ARCHIVE "")
if (ENABLE_SOPHGO AND UNIX AND NOT APPLE AND TARGET libjpeg-turbo)
    set(MD_OCV_JPEG_TURBO_WHOLE_ARCHIVE
        "-Wl,--whole-archive,$<TARGET_FILE:libjpeg-turbo>,--no-whole-archive")
    message(STATUS "Sophgo: force-link bundled libjpeg-turbo (whole-archive) to fix jsimd symbol hijack")
endif ()
