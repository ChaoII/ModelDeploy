add_definitions(-DCATCH_CONFIG_ENABLE_BENCHMARKING)

# Catch2 依赖
if (NOT TARGET Catch2::Catch2WithMain)
    add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/Catch2" "${CMAKE_BINARY_DIR}/third_party/Catch2")
endif ()

add_executable(benchmark benchmark_main.cpp)
target_link_libraries(benchmark PRIVATE ${LIBRARY_NAME} Catch2::Catch2WithMain ${OpenCV_LIBS})
target_include_directories(benchmark PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME benchmark COMMAND benchmark)
set_tests_properties(benchmark PROPERTIES
    ENVIRONMENT "TEST_DATA_DIR=${CMAKE_SOURCE_DIR}")
