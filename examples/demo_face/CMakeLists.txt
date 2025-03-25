
add_executable(demo_face_capi demo_face_capi.cpp)
target_link_libraries(demo_face_capi PUBLIC ${LIBRARY_NAME})


add_executable(demo_face_recognition_cxx demo_face_recognition_cxx.cpp)
target_link_libraries(demo_face_recognition_cxx PUBLIC ${LIBRARY_NAME})
