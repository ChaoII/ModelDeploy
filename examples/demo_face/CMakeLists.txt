
add_executable(demo_face demo_face.cpp)
target_link_libraries(demo_face PUBLIC ${LIB_NAME})
set_target_properties(demo_face PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")