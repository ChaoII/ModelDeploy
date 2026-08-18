
add_executable(demo_pedestrian_attribute_capi demo_pedestrian_attribute_capi.cpp)
target_link_libraries(demo_pedestrian_attribute_capi PUBLIC ${LIBRARY_NAME})
# 本机 360 安全卫士误报并锁定旧输出名 bin\demo_pedestrian_attribute_capi.exe，
# 改用独立输出名规避（仅 Windows；其余平台保持与兄弟示例一致的输出名）。
if(WIN32)
    set_target_properties(demo_pedestrian_attribute_capi PROPERTIES RUNTIME_OUTPUT_NAME demo_pedestrian_attribute_capi2)
endif()

add_executable(demo_pedestrian_attribute_cxx demo_pedestrian_attribute_cxx.cpp)
target_link_libraries(demo_pedestrian_attribute_cxx PUBLIC ${LIBRARY_NAME})

if(ENABLE_SOPHGO)
    add_executable(demo_pedestrian_attribute_sophgo demo_pedestrian_attribute_sophgo.cpp)
    target_link_libraries(demo_pedestrian_attribute_sophgo PUBLIC ${LIBRARY_NAME})

    add_executable(demo_sophgo_clone demo_sophgo_clone.cpp)
    target_link_libraries(demo_sophgo_clone PUBLIC ${LIBRARY_NAME} ${SOPHGO_LIBS})
endif()