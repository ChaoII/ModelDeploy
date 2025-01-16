
add_executable(demo_ocr_common demo_ocr_common.cpp)
target_link_libraries(demo_ocr_common PUBLIC ${LIB_NAME})
set_target_properties(demo_ocr_common PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(demo_ocr_position demo_ocr_position.cpp)
target_link_libraries(demo_ocr_position PUBLIC ${LIB_NAME})
set_target_properties(demo_ocr_position PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")