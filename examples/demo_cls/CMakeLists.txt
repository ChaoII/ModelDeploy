md_add_demo_matrix(classification ${CMAKE_CURRENT_LIST_DIR})

if(ENABLE_SOPHGO)
  add_executable(demo_classification_sophgo demo_classification_sophgo.cpp)
  target_link_libraries(demo_classification_sophgo PUBLIC ${LIBRARY_NAME})
endif()

if(BUILD_CAPI)
  add_executable(demo_classification_capi demo_classification_capi.cpp)
  target_link_libraries(demo_classification_capi PUBLIC ${LIBRARY_NAME})
endif()
