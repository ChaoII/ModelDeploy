//
// Created by aichao on 2025/2/21.
//


#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "csrc/core/md_tensor.h"


namespace modeldeploy::vision {
    /*! @brief Processor for cast images with given type deafault is float.
     */
    class Cast {
    public:
        explicit Cast(const std::string& dtype = "float") : dtype_(dtype) {
        }

        bool ImplByOpenCV(cv::Mat* mat);


        bool operator()(cv::Mat* mat);

        std::string name() { return "Cast"; }
        /** \brief Process the input images
         *
         * \param[in] mat The input image data
         * \param[in] dtype type of data will be casted to
         * \return true if the process successed, otherwise false
         */
        static bool Run(cv::Mat* mat, const std::string& dtype);

        std::string GetDtype() const { return dtype_; }

    private:
        std::string dtype_;
    };
} // namespace modeldeploy::vision
// namespace fastdeploy
