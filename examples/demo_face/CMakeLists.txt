
add_executable(demo_face_capi demo_face_capi.cpp)
target_link_libraries(demo_face_capi PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_face_capi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")