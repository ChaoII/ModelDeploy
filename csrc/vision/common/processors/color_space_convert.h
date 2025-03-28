#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "csrc/core/md_decl.h"


namespace modeldeploy::vision {
    /*! @brief Processor for tansform images from BGR to RGB.
     */
    class MODELDEPLOY_CXX_EXPORT BGR2RGB {
    public:
        bool ImplByOpenCV(cv::Mat* mat);

        virtual std::string Name() { return "BGR2RGB"; }
        bool operator()(cv::Mat* mat);

        /** \brief Process the input images
         *
         * \param[in] mat The input image data
         * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
         * \return true if the process successed, otherwise false
         */
        static bool Run(cv::Mat* mat);
    };

    /*! @brief Processor for tansform images from RGB to BGR.
     */
    class RGB2BGR {
    public:
        bool ImplByOpenCV(cv::Mat* mat);

        std::string Name() { return "RGB2BGR"; }
        bool operator()(cv::Mat* mat);

        /** \brief Process the input images
         *
         * \param[in] mat The input image data
         * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
         * \return true if the process successed, otherwise false
         */
        static bool Run(cv::Mat* mat);
    };

    /*! @brief Processor for tansform images from BGR to GRAY.
     */
    class BGR2GRAY {
    public:
        bool ImplByOpenCV(cv::Mat* mat);

        virtual std::string Name() { return "BGR2GRAY"; }
        bool operator()(cv::Mat* mat);

        /** \brief Process the input images
         *
         * \param[in] mat The input image data
         * \return true if the process successed, otherwise false
         */
        static bool Run(cv::Mat* mat);
    };

    /*! @brief Processor for tansform images from RGB to GRAY.
     */
    class RGB2GRAY {
    public:
        bool ImplByOpenCV(cv::Mat* mat);

        std::string Name() { return "RGB2GRAY"; }
        bool operator()(cv::Mat* mat);

        /** \brief Process the input images
         *
         * \param[in] mat The input image data
         * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
         * \return true if the process successed, otherwise false
         */
        static bool Run(cv::Mat* mat);
    };
} // namespace modeldeploy::vision
// namespace fastdeploy
