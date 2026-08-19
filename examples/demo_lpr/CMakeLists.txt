md_add_demo_matrix(lpr_pipeline ${CMAKE_CURRENT_LIST_DIR})

if(ENABLE_ORT)
  add_executable(demo_lpr_detection_cxx demo_lpr_detection_cxx.cpp)
  target_link_libraries(demo_lpr_detection_cxx PUBLIC ${LIBRARY_NAME})
  add_executable(demo_lpr_recognizer_cxx demo_lpr_recognizer_cxx.cpp)
  target_link_libraries(demo_lpr_recognizer_cxx PUBLIC ${LIBRARY_NAME})
endif()

if(BUILD_CAPI)
  add_executable(demo_lpr_pipeline_capi demo_lpr_pipeline_capi.cpp)
  target_link_libraries(demo_lpr_pipeline_capi PUBLIC ${LIBRARY_NAME})
endif()
