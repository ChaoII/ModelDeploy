#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "csrc/core/md_decl.h"


namespace modeldeploy::vision {
    /*! @brief Processor for transform images from BGR to RGB.
     */
    class MODELDEPLOY_CXX_EXPORT BGR2RGB {
    public:
        bool impl(cv::Mat* mat);

        virtual std::string name() { return "BGR2RGB"; }
        bool operator()(cv::Mat* mat);

        /** \brief Process the input images
         *
         * \param[in] mat The input image data
         * \return true if the process successes, otherwise false
         */
        static bool apply(cv::Mat* mat);
    };

    /*! @brief Processor for tansform images from RGB to BGR.
     */
    class MODELDEPLOY_CXX_EXPORT RGB2BGR {
    public:
        bool impl(cv::Mat* mat);

        std::string name() { return "RGB2BGR"; }
        bool operator()(cv::Mat* mat);

        /** \brief Process the input images
         *
         * \param[in] mat The input image data
         * \return true if the process successed, otherwise false
         */
        static bool apply(cv::Mat* mat);
    };

    /*! @brief Processor for tansform images from BGR to GRAY.
     */
    class MODELDEPLOY_CXX_EXPORT BGR2GRAY {
    public:
        bool impl(cv::Mat* mat);

        virtual std::string name() { return "BGR2GRAY"; }
        bool operator()(cv::Mat* mat);

        /** \brief Process the input images
         *
         * \param[in] mat The input image data
         * \return true if the process successed, otherwise false
         */
        static bool apply(cv::Mat* mat);
    };

    /*! @brief Processor for tansform images from RGB to GRAY.
     */
    class MODELDEPLOY_CXX_EXPORT RGB2GRAY {
    public:
        bool impl(cv::Mat* mat);

        std::string name() { return "RGB2GRAY"; }
        bool operator()(cv::Mat* mat);

        /** \brief Process the input images
         *
         * \param[in] mat The input image data
         * \return true if the process successed, otherwise false
         */
        static bool apply(cv::Mat* mat);
    };
} // namespace modeldeploy::vision
