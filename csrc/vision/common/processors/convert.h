#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "csrc/core/md_decl.h"

namespace modeldeploy::vision {
    /*! @brief Processor for convert images with given paramters.
         */
    class MODELDEPLOY_CXX_EXPORT Convert {
    public:
        Convert(const std::vector<float>& alpha, const std::vector<float>& beta);
        bool operator()(cv::Mat* mat);

        bool ImplByOpenCV(cv::Mat* mat);

        static std::string name() { return "Convert"; }

        // Compute `result = mat * alpha + beta` directly by channel.
        // The default behavior is the same as OpenCV's convertTo method.
        /** \brief Process the input images
             *
             * \param[in] mat The input image data，`result = mat * alpha + beta`
             * \param[in] alpha The alpha channel data
             * \param[in] beta The beta channel data
             * \return true if the process successed, otherwise false
             */
        static bool Run(cv::Mat* mat, const std::vector<float>& alpha,
                        const std::vector<float>& beta);

    private:
        std::vector<float> alpha_;
        std::vector<float> beta_;
    };
}
