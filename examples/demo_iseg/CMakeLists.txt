md_add_demo_matrix(instance_seg ${CMAKE_CURRENT_LIST_DIR})

if(ENABLE_SOPHGO)
  add_executable(demo_iseg_sophgo demo_iseg_sophgo.cpp)
  target_link_libraries(demo_iseg_sophgo PUBLIC ${LIBRARY_NAME})
endif()

if(BUILD_CAPI)
  add_executable(demo_instance_seg_capi demo_instance_seg_capi.cpp)
  target_link_libraries(demo_instance_seg_capi PUBLIC ${LIBRARY_NAME})
endif()
