cmake_minimum_required(VERSION 3.26)
project(demo)

set(CMAKE_CXX_STANDARD 20)


set(LIB_NAME "model_deploy_sdk")
set(SDK_SOURCE "")
set(DEPENDS "")

set(FD_PATH "E:/Fastdeploy/build/install1/FastDeploy.cmake" CACHE FILEPATH "fd cmake file path")
option(BUILD_FACE "build face recognition module" ON)
option(BUILD_ASR "build fun asr module" ON)
option(BUILD_TEST "build unittest" ON)


include(${FD_PATH})
include_directories(${FASTDEPLOY_INCS})
include_directories(${CMAKE_SOURCE_DIR})

message(${FASTDEPLOY_INCS})

list(APPEND DEPENDS ${FASTDEPLOY_LIBS})
add_definitions(-DMODEL_DEPLOY_LIB)
# 添加FastDeploy依赖头文件
file(GLOB_RECURSE DETECTION_SOURCE ${CMAKE_SOURCE_DIR}/src/detection/*.cpp)
file(GLOB_RECURSE OCR_SOURCE ${CMAKE_SOURCE_DIR}/src/ocr/*.cpp)
file(GLOB_RECURSE UTILS_SOURCE ${CMAKE_SOURCE_DIR}/src/utils/*.cpp)
list(APPEND SDK_SOURCE ${DETECTION_SOURCE} ${OCR_SOURCE} ${UTILS_SOURCE})

if (BUILD_FACE)
    set(SEETA_PATH "E:/develop/SeetaFace" CACHE PATH "seeta face directory")
    add_definitions(-DBUILD_FACE)
    include(${CMAKE_SOURCE_DIR}/cmake/face.cmake)
    file(GLOB_RECURSE FACE_SOURCE ${CMAKE_SOURCE_DIR}/src/face/*.cpp)
    list(APPEND SDK_SOURCE ${FACE_SOURCE})
endif ()

if (BUILD_ASR)
    add_definitions(-DBUILD_ASR)
    set(FFMPEG_DIR "E:/develop/ffmpeg-master-latest-win64-gpl-shared" CACHE PATH "ffmpeg directory")
    include(${CMAKE_SOURCE_DIR}/cmake/asr.cmake)
endif ()
set(CMAKE_CXX_STANDARD 20)

add_library(${LIB_NAME} SHARED ${SDK_SOURCE})
target_link_libraries(${LIB_NAME} ${DEPENDS})

if (BUILD_TEST)
    enable_testing()
    add_subdirectory(tests)
endif ()
if (BUILD_ASR)
#    set(CMAKE_CXX_STANDARD 11)
#    SET(RELATION_SOURCE "${CMAKE_SOURCE_DIR}/src/asr/resample.cpp"
#            "${CMAKE_SOURCE_DIR}/src/asr/util.cpp"
#            "${CMAKE_SOURCE_DIR}/src/asr/alignedmem.cpp"
#            "${CMAKE_SOURCE_DIR}/src/asr/encode_converter.cpp")
#    add_executable(demo-asr funasr-onnx-2pass.cpp ${RELATION_SOURCE})
endif ()
add_executable(demo main.cpp)
target_link_libraries(demo ${LIB_NAME})
if (MSVC)
    # 设置控制台输出编码为 UTF-8
    target_compile_definitions(demo PRIVATE _CONSOLE_CP_UTF8)
endif ()
