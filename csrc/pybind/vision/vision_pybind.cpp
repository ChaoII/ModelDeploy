//
// Created by aichao on 2025/6/9.
//

#include <pybind11/pybind11.h>

namespace modeldeploy::vision {
    void bind_vision_struct(const pybind11::module&);
    void bind_classification(const pybind11::module&);
    void bind_ultralytics_det(const pybind11::module&);
    void bind_ultralytics_iseg(const pybind11::module&);
    void bind_ultralytics_obb(const pybind11::module&);
    void bind_ultralytics_pose(const pybind11::module&);
    void bind_lpr_det(const pybind11::module&);
    void bind_lpr_rec(const pybind11::module&);
    void bind_lpr_pipeline(const pybind11::module&);
    void bind_face_det(const pybind11::module&);
    void bind_face_rec(const pybind11::module&);
    void bind_face_age(const pybind11::module&);
    void bind_face_gender(const pybind11::module&);
    void bind_face_as_first(const pybind11::module&);
    void bind_face_as_second(const pybind11::module&);
    void bind_as_pipeline(const pybind11::module&);
    void bind_face_rec_pipeline(const pybind11::module&);
    void bind_visualize(pybind11::module&);
    void bind_ocr_db(const pybind11::module&);
    void bind_ocr_cls(const pybind11::module&);
    void bind_ocr_rec(const pybind11::module&);
    void bind_ocr_pipeline(const pybind11::module&);
    void bind_ocr_layout(const pybind11::module&);
    void bind_ocr_table(const pybind11::module&);
    void bind_table_pipeline(const pybind11::module&);
    void bind_attr(const pybind11::module&);



    void bind_vision(pybind11::module& m) {
        bind_vision_struct(m);
        bind_classification(m);
        bind_ultralytics_det(m);
        bind_ultralytics_iseg(m);
        bind_ultralytics_obb(m);
        bind_ultralytics_pose(m);
        bind_lpr_det(m);
        bind_lpr_rec(m);
        bind_lpr_pipeline(m);
        bind_face_rec(m);
        bind_face_age(m);
        bind_face_gender(m);
        bind_face_as_first(m);
        bind_face_as_second(m);
        bind_as_pipeline(m);
        bind_face_rec_pipeline(m);
        bind_visualize(m);
        bind_ocr_db(m);
        bind_ocr_cls(m);
        bind_ocr_rec(m);
        bind_ocr_pipeline(m);
        bind_ocr_layout(m);
        bind_ocr_table(m);
        bind_table_pipeline(m);
        bind_attr(m);
    }
}
