
#
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

set(mnn_win_x64_static_3_2_0_md_FILE_NAME "mnn_win_x64_static_3_2_0_md.zip")
set(mnn_win_x64_static_3_2_0_mt_FILE_NAME "mnn_win_x64_static_3_2_0_mt.zip")


set(MNN_BASE_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master")
include(FetchContent)

if (WITH_GPU)
    message(FATAL_ERROR "mnn backend GPU is not supported for Windows")
else ()
    if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
        if (WITH_STATIC_CRT)
            set(MNN_FILE_NAME ${mnn_win_x64_static_3_2_0_mt_FILE_NAME})
            set(MNN_URL ${MNN_BASE_URL}/${MNN_FILE_NAME})
            set(MNN_HASH "SHA256=5f89f4bf84643bd196b278d7f23d444dc3b4cb6c464b0e0d1b25cbd4e9099e86")
        else ()
            set(MNN_FILE_NAME ${mnn_win_x64_static_3_2_0_md_FILE_NAME})
            set(MNN_URL ${MNN_BASE_URL}/${MNN_FILE_NAME})
            set(MNN_HASH "SHA256=fc805eae740cf4fe0d2f1c3db301fd0ff4905b4ce5f2ff5b023252e25094fb75")
        endif ()
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
        "${MNN_SOURCE_DIR}/lib/*.dll"
        "${MNN_SOURCE_DIR}/lib/*.so"
        "${MNN_SOURCE_DIR}/lib/*.dylib"
)
file(COPY ${MNN_SHARED_LIBS} DESTINATION ${CMAKE_BINARY_DIR}/bin)





