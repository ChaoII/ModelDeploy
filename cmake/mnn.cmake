
#
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

set(mnn_win_x64_static_3_2_0_md_FILE_NAME "mnn_win_x64_static_3_2_0_md.zip")
set(mnn_win_x64_static_3_2_0_mt_FILE_NAME "mnn_win_x64_static_3_2_0_mt.zip")
set(mnn_linux_x64_static_3_2_0_FILE_NAME "mnn_linux_x64_static_3_2_0.zip")
set(mnn_linux_aarch64_static_3_2_0_FILE_NAME "mnn_linux_aarch64_static_3_2_0.zip")
set(mnn_win_x64_gpu_3_2_0_FILE_NAME "mnn_win_x64_gpu_3_2_0.zip")
set(mnn_linux_x64_gpu_3_2_0_FILE_NAME "mnn_linux_x64_gpu_3_2_0.zip")


set(MNN_BASE_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master")
include(FetchContent)

if (WITH_GPU)
    if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
        set(MNN_FILE_NAME ${mnn_win_x64_gpu_3_2_0_FILE_NAME})
        set(MNN_URL ${MNN_BASE_URL}/${MNN_FILE_NAME})
        set(MNN_HASH "SHA256=ba1404ba39571b7929dd991d8baf67c0eb77ee99ad92629fd9c8f940c8e4c572")

    elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
            set(MNN_FILE_NAME ${mnn_linux_x64_gpu_3_2_0_FILE_NAME})
            set(MNN_URL ${MNN_BASE_URL}/${MNN_FILE_NAME})
            set(MNN_HASH "SHA256=db876f6b347f2adc0aed02ae88dfc5a50a32734b2634ac0d4ab44ecd5eb9be3b")
        else ()
            message(FATAL_ERROR "Unsupported system : ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR} for WITH_GPU=ON")
        endif ()
    else ()
        message(FATAL_ERROR "Unsupported system : ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR} for WITH_GPU=ON")
    endif ()
else ()
    if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /WHOLEARCHIVE:MNN")
        if (WITH_STATIC_CRT)
            set(MNN_FILE_NAME ${mnn_win_x64_static_3_2_0_mt_FILE_NAME})
            set(MNN_URL ${MNN_BASE_URL}/${MNN_FILE_NAME})
            set(MNN_HASH "SHA256=5f89f4bf84643bd196b278d7f23d444dc3b4cb6c464b0e0d1b25cbd4e9099e86")
        else ()
            set(MNN_FILE_NAME ${mnn_win_x64_static_3_2_0_md_FILE_NAME})
            set(MNN_URL ${MNN_BASE_URL}/${MNN_FILE_NAME})
            set(MNN_HASH "SHA256=575e14cfa18a2eaa95c732617393eb424ad75803d7d2162ca985d1d705018b19")
        endif ()
    elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
            set(MNN_FILE_NAME ${mnn_linux_x64_static_3_2_0_FILE_NAME})
            set(MNN_URL ${MNN_BASE_URL}/${MNN_FILE_NAME})
            set(MNN_HASH "SHA256=8e8533220940033da9676c3104e2e299d20f7439fbc102d627aa7855b8d499f3")
        elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
            set(MNN_FILE_NAME ${mnn_linux_aarch64_static_3_2_0_FILE_NAME})
            set(MNN_URL ${MNN_BASE_URL}/${MNN_FILE_NAME})
            set(MNN_HASH "SHA256=d93ab07ec5e19509d79acc4d5bcb1c23284c0834d6deefeadd5e2d57481a31a2")
        endif ()
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--whole-archive -lMNN -Wl,--no-whole-archive")
    else ()
        message(FATAL_ERROR "Unsupported system :" ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR})
    endif ()
endif ()


FetchContent_Declare(mnn
        URL
        ${MNN_URL}
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


set(MNN_LIB_DIR "${mnn_SOURCE_DIR}/lib")
set(MNN_INC_DIR "${mnn_SOURCE_DIR}/include")
include_directories(${MNN_INC_DIR})
link_directories(${MNN_LIB_DIR})


# 拷贝到 ${CMAKE_BINARY_DIR}/bin 或你指定的 bin 目录
file(GLOB MNN_SHARED_LIBS
        "${MNN_LIB_DIR}/*.dll"
        "${MNN_LIB_DIR}/*.so"
        "${MNN_LIB_DIR}/*.dylib"
)
file(COPY ${MNN_SHARED_LIBS} DESTINATION ${CMAKE_BINARY_DIR}/bin)





