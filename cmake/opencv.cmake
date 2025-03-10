#
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")


set(opencv_5_x_win_x64_static_mt_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/opencv_5_x_win_x64_static_mt.zip")
set(opencv_5_x_win_x64_static_md_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/opencv_5_x_win_x64_static_md.zip")
set(opencv_5_x_linux_x64_static_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/opencv_5_x_linux_x64_static.zip")
set(opencv_5_x_linux_aarch64_static_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/opencv_5_x_linux_aarch64_static.zip")
set(opencv_5_x_win_x64_static_mt_HASH "SHA256=86e5bdbce3d2955ef92f295feef1f40171dfdfeac37eaf1b5b141ed8aae3d112")
set(opencv_5_x_win_x64_static_md_HASH "SHA256=587dc46cdb8154cd29e342ff9ad83e5fcaee247aa312f918c5bf1b1372540c38")
set(opencv_5_x_linux_x64_static_HASH "SHA256=4a1ec2ec05b8e8a4d4503e2410543a48dc48beea4a4d3c4832375f6ab7edefb2")
set(opencv_5_x_linux_aarch64_static_HASH "SHA256=5e1cae594bc10fb3ba228478dfaaf975cb7476c2457174dabdc364a44d664fa9")

include(FetchContent)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    if (WITH_STATIC_CRT)
        set(possible_file_locations
                $ENV{HOME}/Downloads/opencv_5_x_win_x64_static_mt_URL.zip
                ${CMAKE_SOURCE_DIR}/opencv_5_x_win_x64_static_mt_URL.zip
                ${CMAKE_BINARY_DIR}/opencv_5_x_win_x64_static_mt_URL.zip
                /tmp/opencv_5_x_win_x64_static_mt_URL.zip
        )
        set(OPENCV_URL "${opencv_5_x_win_x64_static_mt_URL}")
        set(OPENCV_HASH "${opencv_5_x_win_x64_static_mt_HASH}")
    elseif ()
        set(possible_file_locations
                $ENV{HOME}/Downloads/opencv_5_x_win_x64_static_md_URL.zip
                ${CMAKE_SOURCE_DIR}/opencv_5_x_win_x64_static_md_URL.zip
                ${CMAKE_BINARY_DIR}/opencv_5_x_win_x64_static_md_URL.zip
                /tmp/opencv_5_x_win_x64_static_md_URL.zip
        )
        set(OPENCV_URL "${opencv_5_x_win_x64_static_md_URL}")
        set(OPENCV_HASH "${opencv_5_x_win_x64_static_md_HASH}")
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(possible_file_locations
                $ENV{HOME}/Downloads/opencv_5_x_linux_x64_static_URL.zip
                ${CMAKE_SOURCE_DIR}/opencv_5_x_linux_x64_static_URL.zip
                ${CMAKE_BINARY_DIR}/opencv_5_x_linux_x64_static_URL.zip
                /tmp/opencv_5_x_linux_x64_static_URL.zip)
        set(OPENCV_URL "${opencv_5_x_linux_x64_static_URL}")
        set(OPENCV_HASH "${opencv_5_x_linux_x64_static_HASH}")
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(possible_file_locations
                $ENV{HOME}/Downloads/opencv_5_x_linux_aarch64_static_URL.zip
                ${CMAKE_SOURCE_DIR}/opencv_5_x_linux_aarch64_static_URL.zip
                ${CMAKE_BINARY_DIR}/opencv_5_x_linux_aarch64_static_URL.zip
                /tmp/opencv_5_x_linux_aarch64_static_URL.zip)
        set(OPENCV_URL "${opencv_5_x_linux_aarch64_static_URL}")
        set(OPENCV_HASH "${opencv_5_x_linux_aarch64_static_HASH}")
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    message(FATAL_ERROR "Unsupported system :" ${CMAKE_SYSTEM_NAME})
endif ()


foreach (f IN LISTS possible_file_locations)
    if (EXISTS ${f})
        set(OPENCV_URL "${f}")
        file(TO_CMAKE_PATH "${OPENCV_URL}" OPENCV_URL)
        message(STATUS "Found local downloaded opencv: ${OPENCV_URL}")
        break()
    endif ()
endforeach ()

FetchContent_Declare(opencv
        URL
        ${OPENCV_URL}
        URL_HASH ${OPENCV_HASH}
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
    set(opencv_DIR "${opencv_SOURCE_DIR}/x64/vc17/staticlib")
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(opencv_DIR "${opencv_SOURCE_DIR}/lib/cmake/opencv5")
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(opencv_DIR "${opencv_SOURCE_DIR}/lib64/cmake/opencv5")
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    message(FATAL_ERROR "Unsupported system :" ${CMAKE_SYSTEM_NAME})
endif ()

find_package(opencv CONFIG REQUIRED)
