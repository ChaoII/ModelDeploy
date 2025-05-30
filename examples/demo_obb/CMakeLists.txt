

add_executable(demo_obb_cxx demo_obb_cxx.cpp)
target_link_libraries(demo_obb_cxx PUBLIC ${LIBRARY_NAME})


add_executable(demo_obb_capi demo_obb_capi.cpp)
target_link_libraries(demo_obb_capi PUBLIC ${LIBRARY_NAME})