//
// Created by aichao on 2025/6/17.
//

// 如果不加这个头文件，那么std::vector<T> 不会自动转换成python的list
#include <pybind11/stl.h>
#include "pybind/utils/result_convert.h"

namespace modeldeploy::vision {
    Rect2f dict_to_rect2f(const pybind11::dict& box_dict) {
        Rect2f box;
        box.x = box_dict["x"].cast<float>();
        box.y = box_dict["y"].cast<float>();
        box.width = box_dict["width"].cast<float>();
        box.height = box_dict["height"].cast<float>();
        return box;
    }

    pybind11::dict rect2f_to_dict(const Rect2f& box) {
        pybind11::dict box_dict;
        box_dict["x"] = box.x;
        box_dict["y"] = box.y;
        box_dict["width"] = box.width;
        box_dict["height"] = box.height;
        return box_dict;
    }

    RotatedRect dict_to_rotated_rect(const pybind11::dict& box_dict) {
        RotatedRect box;
        box.xc = box_dict["xc"].cast<float>();
        box.yc = box_dict["yc"].cast<float>();
        box.width = box_dict["width"].cast<float>();
        box.height = box_dict["height"].cast<float>();
        box.angle = box_dict["angle"].cast<float>();
        return box;
    }

    pybind11::dict rotated_rect_to_dict(const RotatedRect& box) {
        pybind11::dict box_dict;
        box_dict["xc"] = box.xc;
        box_dict["yc"] = box.yc;
        box_dict["width"] = box.width;
        box_dict["height"] = box.height;
        box_dict["angle"] = box.angle;
        return box_dict;
    }

    Point2f dict_to_point2f(const pybind11::dict& point_dict) {
        Point2f point;
        point.x = point_dict["x"].cast<float>();
        point.y = point_dict["y"].cast<float>();
        return point;
    }

    pybind11::dict point2f_to_dict(const Point2f& point) {
        pybind11::dict point_dict;
        point_dict["x"] = point.x;
        point_dict["y"] = point.y;
        return point_dict;
    }

    Point3f dict_to_point3f(const pybind11::dict& point_dict) {
        Point3f point;
        point.x = point_dict["x"].cast<float>();
        point.y = point_dict["y"].cast<float>();
        point.z = point_dict["z"].cast<float>();
        return point;
    }

    pybind11::dict point3f_to_dict(const Point3f& point) {
        pybind11::dict point_dict;
        point_dict["x"] = point.x;
        point_dict["y"] = point.y;
        point_dict["z"] = point.z;
        return point_dict;
    }

    pybind11::dict mask_to_dict(const Mask& mask) {
        pybind11::dict d;
        d["buffer"] = pybind11::cast(mask.buffer);
        d["shape"] = mask.shape;
        return d;
    }

    // from_dict 工具
    Mask dict_to_mask(const pybind11::dict& d) {
        Mask mask;
        // 反序列化 buffer
        mask.buffer = d["buffer"].cast<std::vector<uint8_t>>();
        // 反序列化 shape
        mask.shape = d["shape"].cast<std::vector<int64_t>>();
        return mask;
    }
}
