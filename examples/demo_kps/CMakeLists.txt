md_add_demo_matrix(pose ${CMAKE_CURRENT_LIST_DIR})

if(ENABLE_ORT)
  add_executable(demo_keypoint_cxx demo_keypoint_cxx.cpp)
  target_link_libraries(demo_keypoint_cxx PUBLIC ${LIBRARY_NAME})
endif()

if(ENABLE_SOPHGO)
  add_executable(demo_pose_sophgo demo_pose_sophgo.cpp)
  target_link_libraries(demo_pose_sophgo PUBLIC ${LIBRARY_NAME})
endif()

if(BUILD_CAPI)
  add_executable(demo_pose_capi demo_pose_capi.cpp)
  target_link_libraries(demo_pose_capi PUBLIC ${LIBRARY_NAME})
endif()
