#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "csrc/core/md_decl.h"


namespace modeldeploy::vision {
    /*! @brief Processor for crop images in center with given type deafault is float.
         */
    class MODELDEPLOY_CXX_EXPORT CenterCrop {
    public:
        CenterCrop(int width, int height) : height_(height), width_(width) {
        }

        bool ImplByOpenCV(cv::Mat* mat);
        static std::string name() { return "CenterCrop"; }
        bool operator()(cv::Mat* mat);

        /** \brief Process the input images
             *
             * \param[in] mat The input image data
             * \param[in] width width of data will be croped to
             * \param[in] height height of data will be croped to
             * \return true if the process successed, otherwise false
             */
        static bool Run(cv::Mat* mat, const int& width, const int& height);

    private:
        int height_;
        int width_;
    };
}
