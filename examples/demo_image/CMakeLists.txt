
add_executable(demo_image_from_base64 demo_image_from_base64.cpp)
target_link_libraries(demo_image_from_base64 PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_image_from_base64 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
