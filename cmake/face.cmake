include_directories(${SEETA_PATH}/include)
if (WIN32)
    link_directories(${SEETA_PATH}/lib/x64)
    list(APPEND DEPENDS
            SeetaFaceLandmarker600
            SeetaFaceRecognizer610
            SeetaFaceDetector600
            SeetaFaceAntiSpoofingX600
            SeetaQualityAssessor300
            SeetaPoseEstimation600
            SeetaGenderPredictor600
            SeetaEyeStateDetector200
            SeetaAgePredictor600)
else (WIN32)
    message(FATAL_ERROR "Unsupported OS")
endif ()
