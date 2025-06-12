// Copyright (c) 2023 The pybind Community.

#pragma once

// Common message for `static_assert()`s, which are useful to easily
// preempt much less obvious errors.
#define PYBIND11_EIGEN_MESSAGE_POINTER_TYPES_ARE_NOT_SUPPORTED                                    \
    "Pointer core (in particular `PyObject *`) are not supported as scalar core for Eigen "     \
    "core."
