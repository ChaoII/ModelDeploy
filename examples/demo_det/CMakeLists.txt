# 主模型矩阵（demo_detection_{ort_cpu,ort_gpu_cuda_ep,ort_gpu_trt_ep,mnn_*,sophgo_tpu_*}）
md_add_demo_matrix(detection ${CMAKE_CURRENT_LIST_DIR})

# 保留特殊 demo
if(ENABLE_ORT)
  add_executable(demo_detection_batch demo_detection_batch.cpp)
  target_link_libraries(demo_detection_batch PUBLIC ${LIBRARY_NAME})
  add_executable(demo_detection_multi_thread demo_detection_multi_thread.cpp)
  target_link_libraries(demo_detection_multi_thread PUBLIC ${LIBRARY_NAME})
  add_executable(demo_detection_multi_thread_trt demo_detection_multi_thread_trt.cpp)
  target_link_libraries(demo_detection_multi_thread_trt PUBLIC ${LIBRARY_NAME})
  add_executable(demo_multi_thread_compare demo_multi_thread_compare.cpp)
  target_link_libraries(demo_multi_thread_compare PUBLIC ${LIBRARY_NAME})
  add_executable(demo_benchmark demo_benchmark.cpp)
  target_link_libraries(demo_benchmark PUBLIC ${LIBRARY_NAME})
  add_executable(demo_profile demo_profile.cpp)
  target_link_libraries(demo_profile PUBLIC ${LIBRARY_NAME})
endif()

if(BUILD_CAPI)
  add_executable(demo_detection_capi demo_detection_capi.cpp)
  target_link_libraries(demo_detection_capi PUBLIC ${LIBRARY_NAME})
endif()

if(ENABLE_SOPHGO)
  add_executable(demo_detection_sophgo demo_detection_sophgo.cpp)
  target_link_libraries(demo_detection_sophgo PUBLIC ${LIBRARY_NAME})
endif()
