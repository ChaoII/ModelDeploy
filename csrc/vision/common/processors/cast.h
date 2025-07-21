//
// Created by aichao on 2025/2/21.
//


#pragma once

#include <string>
#include <vector>
#include "core/tensor.h"
#include "vision/common/image_data.h"


namespace modeldeploy::vision {
    /*! @brief Processor for cast images with given type default is float.
     */
    class MODELDEPLOY_CXX_EXPORT Cast {
    public:
        explicit Cast(const std::string& dtype = "float") : dtype_(dtype) {
        }

        bool impl(ImageData* image) const;

        bool operator()(ImageData* image) const;

        std::string name() { return "Cast"; }
        /** \brief Process the input images
         *
         * \param[in] image The input image data
         * \param[in] dtype type of data will be casted to
         * \return true if the process successfully, otherwise false
         */
        static bool apply(ImageData* image, const std::string& dtype);

        std::string get_dtype() const { return dtype_; }

    private:
        std::string dtype_;
    };
} // namespace modeldeploy::vision
