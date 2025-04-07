add_executable(demo_face_det_cxx demo_face_det_cxx.cpp)
target_link_libraries(demo_face_det_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_face_det_capi demo_face_det_capi.cpp)
target_link_libraries(demo_face_det_capi PUBLIC ${LIBRARY_NAME})



add_executable(demo_face_age_cxx demo_face_age_cxx.cpp)
target_link_libraries(demo_face_age_cxx PUBLIC ${LIBRARY_NAME})


add_executable(demo_face_gender_cxx demo_face_gender_cxx.cpp)
target_link_libraries(demo_face_gender_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_face_rec_cxx demo_face_rec_cxx.cpp)
target_link_libraries(demo_face_rec_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_face_rec_capi demo_face_rec_capi.cpp)
target_link_libraries(demo_face_rec_capi PUBLIC ${LIBRARY_NAME})


add_executable(demo_face_pipeline_cxx demo_face_rec_pipeline_cxx.cpp)
target_link_libraries(demo_face_pipeline_cxx PUBLIC ${LIBRARY_NAME})


add_executable(demo_face_as_first_cxx demo_face_as_first_cxx.cpp)
target_link_libraries(demo_face_as_first_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_face_as_second_cxx demo_face_as_second_cxx.cpp)
target_link_libraries(demo_face_as_second_cxx PUBLIC ${LIBRARY_NAME})


add_executable(demo_face_as_pipeline_cxx demo_face_as_pipeline_cxx.cpp)
target_link_libraries(demo_face_as_pipeline_cxx PUBLIC ${LIBRARY_NAME})
