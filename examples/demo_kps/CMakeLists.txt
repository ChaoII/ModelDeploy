

add_executable(demo_pose_cxx demo_pose_cxx.cpp)
target_link_libraries(demo_pose_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_keypoint_cxx demo_keypoint_cxx.cpp)
target_link_libraries(demo_keypoint_cxx PUBLIC ${LIBRARY_NAME})

# Sophgo TPU 后端 demo
if(ENABLE_SOPHGO)
add_executable(demo_pose_sophgo demo_pose_sophgo.cpp)
target_link_libraries(demo_pose_sophgo PUBLIC ${LIBRARY_NAME})
endif()

add_executable(demo_pose_capi demo_pose_capi.cpp)
target_link_libraries(demo_pose_capi PUBLIC ${LIBRARY_NAME})