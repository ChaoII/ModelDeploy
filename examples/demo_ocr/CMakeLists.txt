
add_executable(demo_ocr_cxx demo_ocr_cxx.cpp)
target_link_libraries(demo_ocr_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_ocr_capi demo_ocr_capi.cpp)
target_link_libraries(demo_ocr_capi PUBLIC ${LIBRARY_NAME})

add_executable(demo_ocr_recognition_capi demo_ocr_recognition_capi.cpp)
target_link_libraries(demo_ocr_recognition_capi PUBLIC ${LIBRARY_NAME})

add_executable(demo_ocr_recognition_capi_batch demo_ocr_recognition_capi_batch.cpp)
target_link_libraries(demo_ocr_recognition_capi_batch PUBLIC ${LIBRARY_NAME})

add_executable(demo_structure_table_cxx demo_structure_table_cxx.cpp)
target_link_libraries(demo_structure_table_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_pp_structure_table_cxx demo_pp_structure_table_cxx.cpp)
target_link_libraries(demo_pp_structure_table_cxx PUBLIC ${LIBRARY_NAME})