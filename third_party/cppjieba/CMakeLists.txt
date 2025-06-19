project(cppjieba)

cmake_minimum_required(VERSION 3.15)

include_directories(${PROJECT_SOURCE_DIR}/deps/limonp/include
  ${PROJECT_SOURCE_DIR}/include)

add_library(cppjieba INTERFACE)
target_include_directories(cppjieba INTERFACE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/deps/limonp/include
)
