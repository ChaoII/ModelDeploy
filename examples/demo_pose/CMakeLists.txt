

add_executable(demo_pose_cxx demo_pose_cxx.cpp)
target_link_libraries(demo_pose_cxx PUBLIC ${LIBRARY_NAME})


add_executable(demo_pose_capi demo_pose_capi.cpp)
target_link_libraries(demo_pose_capi PUBLIC ${LIBRARY_NAME})