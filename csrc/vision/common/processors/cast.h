//
// Created by aichao on 2025/2/21.
//


#pragma once

#include <string>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>
#include "core/tensor.h"
#include "vision/common/image_data.h"


namespace modeldeploy::vision {
    /*! @brief Processor for cast images with given type default is float.
     */
    class MODELDEPLOY_CXX_EXPORT Cast {
    public:
        explicit Cast(std::string  dtype = "float") : dtype_(std::move(dtype)) {
        }

        bool impl(cv::Mat* mat) const;

        bool operator()(cv::Mat* mat) const;

        static std::string name() { return "Cast"; }
        /** \brief Process the input images
         *
         * \param[in] mat The input image data
         * \param[in] dtype type of data will be casted to
         * \return true if the process successfully, otherwise false
         */
        static bool apply(cv::Mat* mat, const std::string& dtype);

        [[nodiscard]] std::string get_dtype() const { return dtype_; }

    private:
        std::string dtype_;
    };
} // namespace modeldeploy::vision
