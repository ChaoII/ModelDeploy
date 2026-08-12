
add_executable(demo_classification_cxx demo_classification_cxx.cpp)
target_link_libraries(demo_classification_cxx PUBLIC ${LIBRARY_NAME})

# Sophgo TPU 后端 demo
if(ENABLE_SOPHGO)
add_executable(demo_classification_sophgo demo_classification_sophgo.cpp)
target_link_libraries(demo_classification_sophgo PUBLIC ${LIBRARY_NAME})
endif()

add_executable(demo_classification_capi demo_classification_capi.cpp)
target_link_libraries(demo_classification_capi PUBLIC ${LIBRARY_NAME})