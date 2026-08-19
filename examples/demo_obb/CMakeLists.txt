md_add_demo_matrix(obb ${CMAKE_CURRENT_LIST_DIR})

if(ENABLE_SOPHGO)
  add_executable(demo_obb_sophgo demo_obb_sophgo.cpp)
  target_link_libraries(demo_obb_sophgo PUBLIC ${LIBRARY_NAME})
endif()

if(BUILD_CAPI)
  add_executable(demo_obb_capi demo_obb_capi.cpp)
  target_link_libraries(demo_obb_capi PUBLIC ${LIBRARY_NAME})
endif()
