//
// Created by aichao on 2025/2/8.
//

#pragma once

#ifdef _WIN32
#define strdup _strdup
#endif

#define MD_FACE_DETECT             (1 << 0)
#define MD_FACE_LANDMARK           (1 << 1)
#define MD_FACE_RECOGNITION        (1 << 2)
#define MD_FACE_ANTI_SPOOfING      (1 << 3)
#define MD_FACE_QUALITY_EVALUATE   (1 << 4)
#define MD_FACE_AGE_ATTRIBUTE      (1 << 5)
#define MD_FACE_GENDER_ATTRIBUTE   (1 << 6)
#define MD_FACE_EYE_STATE          (1 << 7)


#define MD_MASK (MD_FACE_DETECT | MD_FACE_LANDMARK | MD_FACE_RECOGNITION \
                |MD_FACE_ANTI_SPOOfING | MD_FACE_QUALITY_EVALUATE | MD_FACE_AGE_ATTRIBUTE \
                |MD_FACE_GENDER_ATTRIBUTE | MD_FACE_EYE_STATE)
