

add_executable(demo_obb_cxx demo_obb_cxx.cpp)
target_link_libraries(demo_obb_cxx PUBLIC ${LIBRARY_NAME})

# Sophgo TPU 后端 demo
if(ENABLE_SOPHGO)
add_executable(demo_obb_sophgo demo_obb_sophgo.cpp)
target_link_libraries(demo_obb_sophgo PUBLIC ${LIBRARY_NAME})
endif()

add_executable(demo_obb_capi demo_obb_capi.cpp)
target_link_libraries(demo_obb_capi PUBLIC ${LIBRARY_NAME})