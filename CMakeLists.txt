cmake_minimum_required(VERSION 3.26)
project(demo)

set(CMAKE_CXX_STANDARD 17)

if (MSVC)
    # 设置源文件编码为 UTF-8
    add_compile_options(/utf-8)
endif ()


set(LIB_NAME "model_deploy_sdk")
set(SDK_SOURCE "")
set(DEPENDS "")

set(FD_PATH "E:/Fastdeploy/build/install1/FastDeploy.cmake" CACHE FILEPATH "fd cmake file path")
option(BUILD_FACE "build face recognition module" ON)
option(BUILD_TEST "build unittest" ON)

include(${FD_PATH})
include_directories(${FASTDEPLOY_INCS})
include_directories(${CMAKE_SOURCE_DIR})

list(APPEND ${FASTDEPLOY_LIBS})


add_definitions(-DMODEL_DEPLOY_LIB)
# 添加FastDeploy依赖头文件
file(GLOB SDK_SOURCE ${CMAKE_SOURCE_DIR}/src/*.cpp)
if (BUILD_FACE)
    set(SEETA_PATH "E:/develop/SeetaFace" CACHE PATH "seeta face directory")
    include_directories(${SEETA_PATH}/include)
    link_directories((${SEETA_PATH}/lib/x64))
    list(APPEND DEPENDS
            SeetaFaceLandmarker600
            SeetaFaceRecognizer610
            SeetaFaceDetector600
            SeetaFaceAntiSpoofingX600)
    file(GLOB FACE_SOURCE ${CMAKE_SOURCE_DIR}/src/face/*.cpp)
    list(APPEND SDK_SOURCE ${FACE_SOURCE})
endif ()


add_library(${LIB_NAME} SHARED ${SDK_SOURCE})
target_link_libraries(${LIB_NAME} ${DEPENDS})

if (BUILD_TEST)
    enable_testing()
    add_subdirectory(tests)
endif ()

add_executable(demo main.cpp)
target_link_libraries(demo ${LIB_NAME})
if (MSVC)
    # 设置控制台输出编码为 UTF-8
    target_compile_definitions(demo PRIVATE _CONSOLE_CP_UTF8)
endif ()
