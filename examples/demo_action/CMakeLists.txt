# 动作识别 demo_action（TSN，VideoDecoder 抽帧） / demo_action_skeleton（ST-GCN）需要 BUILD_VISION；
# 骨架 demo 额外需要 BUILD_VIDEO（VideoDecoder + FFmpeg）。
if (BUILD_VISION)
  add_executable(demo_action demo_action.cpp)
  target_link_libraries(demo_action PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
  if (BUILD_VIDEO)
    add_executable(demo_action_skeleton demo_action_skeleton.cpp)
    target_link_libraries(demo_action_skeleton PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
  endif ()
endif ()
