md_add_demo_matrix(sem ${CMAKE_CURRENT_LIST_DIR})

if(ENABLE_SOPHGO)
  add_executable(demo_sem_sophgo demo_sem_sophgo.cpp)
  target_link_libraries(demo_sem_sophgo PUBLIC ${LIBRARY_NAME})
endif()
