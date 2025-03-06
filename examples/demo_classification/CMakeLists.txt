
add_executable(demo_classification demo_classification.cpp)
target_link_libraries(demo_classification PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_classification PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")