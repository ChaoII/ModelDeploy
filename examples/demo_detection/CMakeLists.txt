
add_executable(demo_detection_cxx demo_detection_cxx.cpp)
target_link_libraries(demo_detection_cxx PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_detection_cxx PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(demo_detection_capi demo_detection_capi.cpp)
target_link_libraries(demo_detection_capi PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_detection_capi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")