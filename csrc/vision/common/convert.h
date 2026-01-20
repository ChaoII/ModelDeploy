//
// Created by aichao on 2026/1/19.
//

#pragma once
#include "vision/common/basic_types.h"

namespace modeldeploy::vision {
    int md_image_type_to_ocv_type(MdImageType type);
    MdImageType md_image_type_from_ocv_type(int ocv_type);
    int md_color_convert_type_to_ocv_color_convert_type(ColorConvertType md_type);
}
