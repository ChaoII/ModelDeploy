add_definitions(-DCATCH_CONFIG_ENABLE_BENCHMARKING)

add_executable(benchmark_image_data benchmark_image_data.cpp)
target_link_libraries(benchmark_image_data PRIVATE ${LIBRARY_NAME} Catch2::Catch2WithMain ${OpenCV_LIBS})
add_test(NAME benchmark_image_data COMMAND benchmark_image_data)
set_tests_properties(benchmark_image_data PROPERTIES ENVIRONMENT "TEST_DATA_DIR=${CMAKE_SOURCE_DIR}")

if (BUILD_VISION)
    add_executable(benchmark_yolo_preproc benchmark_yolo_preproc.cpp)
    target_link_libraries(benchmark_yolo_preproc PRIVATE ${LIBRARY_NAME} Catch2::Catch2WithMain ${OpenCV_LIBS})
    add_test(NAME benchmark_yolo_preproc COMMAND benchmark_yolo_preproc)
    set_tests_properties(benchmark_yolo_preproc PROPERTIES ENVIRONMENT "TEST_DATA_DIR=${CMAKE_SOURCE_DIR}")
endif ()
