//
// Created by aichao on 2025/6/17.
//

#pragma once
#include <pybind11/pybind11.h>
#include <csrc/vision/common/struct.h>
#include <csrc/vision/common/result.h>

namespace modeldeploy::vision {
    Rect2f dict_to_rect2f(const pybind11::dict& box_dict);

    pybind11::dict rect2f_to_dict(const Rect2f& box);

    RotatedRect dict_to_rotated_rect(const pybind11::dict& box_dict);

    pybind11::dict rotated_rect_to_dict(const RotatedRect& box);

    Point2f dict_to_point2f(const pybind11::dict& point_dict);

    pybind11::dict point2f_to_dict(const Point2f& point);

    Point3f dict_to_point3f(const pybind11::dict& point_dict);

    pybind11::dict point3f_to_dict(const Point3f& point);

    Mask dict_to_mask(const pybind11::dict& d);

    pybind11::dict mask_to_dict(const Mask& mask);
}
