cmake_minimum_required(VERSION 3.26)
project(demo)

set(CMAKE_CXX_STANDARD 17)
set(LIB_NAME "model_deploy_sdk")

option(BUILD_TEST "build unittest" ON)

include(E:/Fastdeploy/build/install1/FastDeploy.cmake)
if (MSVC)
    # 设置源文件编码为 UTF-8
    add_compile_options(/utf-8)
endif ()

add_definitions(-DMODEL_DEPLOY_LIB)
# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})
include_directories(${CMAKE_SOURCE_DIR})

file(GLOB SDK_SOURCE ${CMAKE_SOURCE_DIR}/src/*.cpp)


add_library(${LIB_NAME} SHARED ${SDK_SOURCE})
target_link_libraries(${LIB_NAME} ${FASTDEPLOY_LIBS})

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
