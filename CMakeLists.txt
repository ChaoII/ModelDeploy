cmake_minimum_required(VERSION 3.26)
project(model_deploy_sdk)

set(CMAKE_CXX_STANDARD 17)

set(LIB_NAME "model_deploy_sdk")
set(SDK_SOURCE "")
set(DEPENDS "")

option(BUILD_DETECTION "build face detection module" ON)
option(BUILD_OCR "build face ocr module" ON)
option(BUILD_FACE "build face recognition module" ON)
option(BUILD_ASR "build asr module" ON)
option(BUILD_ASR_WSS_SERVER "build asr websocket server" OFF)
option(BUILD_EXAMPLES "build examples" ON)
option(WITH_SAVE_LOG "save log to log file" OFF)
option(BUILD_TEST "build unittest" OFF)
set(FD_DIR "E:/Fastdeploy/build/install_gpu" CACHE PATH "fd cmake file path")


if (MSVC)
    add_link_options("/MANIFEST:NO")
    add_compile_options(/wd4291 /wd4305 /wd4244 /wd4828 /wd4251 /wd4275)
endif ()


if (WITH_SAVE_LOG)
    add_definitions(-DSAVE_LOG)
endif ()
set(BUILD_SHARED_LIBS OFF)
add_subdirectory("${CMAKE_SOURCE_DIR}/3rd_party/spdlog")
list(APPEND DEPENDS spdlog::spdlog)

include(${FD_DIR}/FastDeploy.cmake)
include_directories(${FASTDEPLOY_INCS})
include_directories(${CMAKE_SOURCE_DIR})

list(APPEND DEPENDS ${FASTDEPLOY_LIBS})
add_definitions(-DMODEL_DEPLOY_LIB)

# 添加FastDeploy依赖头文件
file(GLOB_RECURSE UTILS_SOURCE ${CMAKE_SOURCE_DIR}/csrc/utils/*.cpp)
list(APPEND SDK_SOURCE ${UTILS_SOURCE})

if (BUILD_DETECTION)
    add_definitions(-DBUILD_DETECTION)
    file(GLOB_RECURSE DETECTION_SOURCE ${CMAKE_SOURCE_DIR}/csrc/vision/detection/*.cpp)
    list(APPEND SDK_SOURCE ${DETECTION_SOURCE})
endif ()

if (BUILD_OCR)
    add_definitions(-DBUILD_OCR)
    file(GLOB_RECURSE OCR_SOURCE ${CMAKE_SOURCE_DIR}/csrc/vision/ocr/*.cpp)
    list(APPEND SDK_SOURCE ${OCR_SOURCE})
endif ()


if (BUILD_FACE)
    set(SEETA_DIR "E:/develop/SeetaFace" CACHE PATH "seeta face directory")
    add_definitions(-DBUILD_FACE)
    include(${CMAKE_SOURCE_DIR}/cmake/face.cmake)
    file(GLOB_RECURSE FACE_SOURCE ${CMAKE_SOURCE_DIR}/csrc/vision/face/*.cpp)
    list(APPEND SDK_SOURCE ${FACE_SOURCE})
endif ()

if (BUILD_ASR_WSS_SERVER)
    if (NOT BUILD_ASR)
        message(WARNING "build asr websocket server must set BUILD_ASR ON, set BUILD_ASR ON")
        set(BUILD_ASR ON)
    endif ()
    add_subdirectory(${CMAKE_SOURCE_DIR}/csrc/audio/asr/bin)
endif ()


if (BUILD_ASR)
    add_definitions(-DBUILD_ASR)
    set(FFMPEG_DIR "E:/develop/ffmpeg-master-latest-win64-gpl-shared" CACHE PATH "ffmpeg directory")
    include(${CMAKE_SOURCE_DIR}/cmake/asr.cmake)
    file(GLOB ASR_SOURCE ${CMAKE_SOURCE_DIR}/csrc/audio/asr/*.cpp)
    list(APPEND SDK_SOURCE ${ASR_SOURCE})
endif ()


add_library(${LIB_NAME} SHARED ${SDK_SOURCE})
target_link_libraries(${LIB_NAME} PUBLIC ${DEPENDS})

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (BUILD_TEST)
    enable_testing()
    add_subdirectory(tests)
endif ()


