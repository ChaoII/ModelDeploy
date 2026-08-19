md_add_demo_matrix(depth ${CMAKE_CURRENT_LIST_DIR})

if(ENABLE_SOPHGO)
  add_executable(demo_depth_sophgo demo_depth_sophgo.cpp)
  target_link_libraries(demo_depth_sophgo PUBLIC ${LIBRARY_NAME})
endif()
