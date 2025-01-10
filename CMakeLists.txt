cmake_minimum_required(VERSION 3.26)
project(demo)

set(CMAKE_CXX_STANDARD 17)

set(LIB_NAME "model_deploy_sdk")
set(SDK_SOURCE "")
set(DEPENDS "")

option(BUILD_FACE "build face recognition module" ON)
option(BUILD_ASR "build fun asr module" ON)
option(BUILD_TEST "build unittest" OFF)
set(FD_DIR "E:/Fastdeploy/build/install1" CACHE PATH "fd cmake file path")

include(${FD_DIR}/FastDeploy.cmake)
include_directories(${FASTDEPLOY_INCS})
include_directories(${CMAKE_SOURCE_DIR})

list(APPEND DEPENDS ${FASTDEPLOY_LIBS})
add_definitions(-DMODEL_DEPLOY_LIB)
# 添加FastDeploy依赖头文件
file(GLOB_RECURSE DETECTION_SOURCE ${CMAKE_SOURCE_DIR}/src/detection/*.cpp)
file(GLOB_RECURSE OCR_SOURCE ${CMAKE_SOURCE_DIR}/src/ocr/*.cpp)
file(GLOB_RECURSE UTILS_SOURCE ${CMAKE_SOURCE_DIR}/src/utils/*.cpp)
list(APPEND SDK_SOURCE ${DETECTION_SOURCE} ${OCR_SOURCE} ${UTILS_SOURCE})

if (BUILD_FACE)
    set(SEETA_DIR "E:/develop/SeetaFace" CACHE PATH "seeta face directory")
    add_definitions(-DBUILD_FACE)
    include(${CMAKE_SOURCE_DIR}/cmake/face.cmake)
    file(GLOB_RECURSE FACE_SOURCE ${CMAKE_SOURCE_DIR}/src/face/*.cpp)
    list(APPEND SDK_SOURCE ${FACE_SOURCE})
endif ()

if (BUILD_ASR)
    add_definitions(-DBUILD_ASR)
    set(FFMPEG_DIR "E:/develop/ffmpeg-master-latest-win64-gpl-shared" CACHE PATH "ffmpeg directory")
    include(${CMAKE_SOURCE_DIR}/cmake/asr.cmake)
    file(GLOB ASR_SOURCE ${CMAKE_SOURCE_DIR}/src/asr/*.cpp)
    list(APPEND SDK_SOURCE ${ASR_SOURCE})
endif ()

add_library(${LIB_NAME} SHARED ${SDK_SOURCE})
target_link_libraries(${LIB_NAME} PUBLIC ${DEPENDS})

if (BUILD_TEST)
    enable_testing()
    add_subdirectory(tests)
endif ()



include("${CMAKE_SOURCE_DIR}/cmake/asr_wss_server.cmake")


#add_executable(demo main.cpp)
#target_link_libraries(demo ${LIB_NAME})

