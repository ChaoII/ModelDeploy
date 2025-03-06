//
// Created by aichao on 2025/2/8.
//

#pragma once

#ifdef _WIN32
#define strdup _strdup
#endif

#define MD_FACE_DETECT             (0b00000001)
#define MD_FACE_LANDMARK           (0b00000001 << 1)
#define MD_FACE_RECOGNITION        (0b00000001 << 2)
#define MD_FACE_ANTI_SPOOfING      (0b00000001 << 3)
#define MD_FACE_QUALITY_EVALUATE   (0b00000001 << 4)
#define MD_FACE_AGE_ATTRIBUTE      (0b00000001 << 5)
#define MD_FACE_GENDER_ATTRIBUTE   (0b00000001 << 6)
#define MD_FACE_EYE_STATE          (0b00000001 << 7)


#define MD_MASK (MD_FACE_DETECT | MD_FACE_LANDMARK | MD_FACE_RECOGNITION \
                |MD_FACE_ANTI_SPOOfING | MD_FACE_QUALITY_EVALUATE | MD_FACE_AGE_ATTRIBUTE \
                |MD_FACE_GENDER_ATTRIBUTE | MD_FACE_EYE_STATE)

