if(BUILD_VISION)
  add_executable(demo_doc demo_doc.cpp)
  target_link_libraries(demo_doc PRIVATE ${LIBRARY_NAME})
endif()
