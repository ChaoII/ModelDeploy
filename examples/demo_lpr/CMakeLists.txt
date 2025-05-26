
add_executable(demo_lpr_detection_cxx demo_lpr_detection_cxx.cpp)
target_link_libraries(demo_lpr_detection_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_lpr_recognizer_cxx demo_lpr_recognizer_cxx.cpp)
target_link_libraries(demo_lpr_recognizer_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_lpr_pipeline_cxx demo_lpr_pipeline_cxx.cpp)
target_link_libraries(demo_lpr_pipeline_cxx PUBLIC ${LIBRARY_NAME})


#---------------------------------------capi-------------------------------

add_executable(demo_lpr_detection_capi demo_lpr_detection_capi.cpp)
target_link_libraries(demo_lpr_detection_capi PUBLIC ${LIBRARY_NAME})

add_executable(demo_lpr_recognizer_capi demo_lpr_recognizer_capi.cpp)
target_link_libraries(demo_lpr_recognizer_capi PUBLIC ${LIBRARY_NAME})

add_executable(demo_lpr_pipeline_capi demo_lpr_pipeline_capi.cpp)
target_link_libraries(demo_lpr_pipeline_capi PUBLIC ${LIBRARY_NAME})