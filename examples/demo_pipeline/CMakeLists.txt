md_add_demo_matrix(pedestrian_attribute ${CMAKE_CURRENT_LIST_DIR})

if(ENABLE_ORT)
  add_executable(demo_pedestrian_attribute demo_pedestrian_attribute.cpp)
  target_link_libraries(demo_pedestrian_attribute PUBLIC ${LIBRARY_NAME})
endif()

if(BUILD_CAPI)
  add_executable(demo_pedestrian_attribute_capi demo_pedestrian_attribute_capi.cpp)
  target_link_libraries(demo_pedestrian_attribute_capi PUBLIC ${LIBRARY_NAME})
  # 兼容 360 安全卫士误报：输出改名为 demo_pedestrian_attribute_capi.exe
  if(WIN32)
    set_target_properties(demo_pedestrian_attribute_capi PROPERTIES RUNTIME_OUTPUT_NAME demo_pedestrian_attribute_capi)
  endif()
endif()

if(ENABLE_SOPHGO)
  add_executable(demo_pedestrian_attribute_sophgo demo_pedestrian_attribute_sophgo.cpp)
  target_link_libraries(demo_pedestrian_attribute_sophgo PUBLIC ${LIBRARY_NAME})

  add_executable(demo_sophgo_clone demo_sophgo_clone.cpp)
  target_link_libraries(demo_sophgo_clone PUBLIC ${LIBRARY_NAME} ${SOPHGO_LIBS})
endif()
