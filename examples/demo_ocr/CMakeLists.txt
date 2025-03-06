
add_executable(demo_ocr_cxx demo_ocr_cxx.cpp)
target_link_libraries(demo_ocr_cxx PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_ocr_cxx PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")


add_executable(demo_ocr_capi demo_ocr_capi.cpp)
target_link_libraries(demo_ocr_capi PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_ocr_capi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(demo_ocr_recognition_capi demo_ocr_recognition_capi.cpp)
target_link_libraries(demo_ocr_recognition_capi PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_ocr_recognition_capi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(demo_ocr_recognition_capi_batch demo_ocr_recognition_capi_batch.cpp)
target_link_libraries(demo_ocr_recognition_capi_batch PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_ocr_recognition_capi_batch PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
