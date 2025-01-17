add_executable(demo_utils demo_utils.cpp)
target_link_libraries(demo_utils PUBLIC ${LIB_NAME})
set_target_properties(demo_utils PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")