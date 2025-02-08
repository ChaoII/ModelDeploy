
add_executable(demo_detection demo_detection.cpp)
target_link_libraries(demo_detection PUBLIC ${LIB_NAME})
set_target_properties(demo_detection PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")