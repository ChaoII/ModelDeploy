//
// Created by aichao on 2025/2/21.
//


#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include "csrc/core/md_decl.h"
#include "csrc/core/tensor.h"

namespace modeldeploy::vision {
    /*! @brief Processor for transform images from HWC to CHW.
    */
    class MODELDEPLOY_CXX_EXPORT HWC2CHW {
    public:
        bool impl(cv::Mat* mat);

        std::string name() { return "HWC2CHW"; }

        bool operator()(cv::Mat* mat);

        /** \brief Process the input images
         *
         * \param[in] mat The input image data
         * \return true if the process successfully, otherwise false
         */
        static bool apply(cv::Mat* mat);
    };
}
