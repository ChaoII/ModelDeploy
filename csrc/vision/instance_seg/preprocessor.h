//
// Created by aichao on 2025/4/14.
//

#pragma once
#include "csrc/vision/common/result.h"
#include "csrc/core/md_tensor.h"

namespace modeldeploy::vision::detection {
    /*! @brief Preprocessor object for YOLOv5Seg serials model.
   */
    class MODELDEPLOY_CXX_EXPORT YOLOv5SegPreprocessor {
    public:
        /// Create a preprocessor instance for YOLOv5Seg serials model
        YOLOv5SegPreprocessor();
        /** \brief Process the input image and prepare input tensors for runtime
        *
        * \param[in] images The input image data list, all the elements are returned by cv::imread()
        * \param[in] outputs The output tensors which will feed in runtime
        * \param[in] ims_info The shape info list, record input_shape and output_shape
        * \return true if the preprocess successed, otherwise false
        */
        bool run(std::vector<cv::Mat>* images, std::vector<MDTensor>* outputs,
                 std::vector<std::map<std::string, std::array<float, 2>>>* ims_info);

        /// Set target size, tuple of (width, height), default size = {640, 640}
        void set_size(const std::vector<int>& size) { size_ = size; }

        /// Get target size, tuple of (width, height), default size = {640, 640}
        [[nodiscard]] std::vector<int> GetSize() const { return size_; }

        /// Set padding value, size should be the same as channels
        void set_padding_value(const std::vector<float>& padding_value) {
            padding_value_ = padding_value;
        }

        /// Get padding value, size should be the same as channels
        [[nodiscard]] std::vector<float> get_padding_value() const { return padding_value_; }

        /// Set is_scale_up, if is_scale_up is false, the input image only
      /// can be zoom out, the maximum resize scale cannot exceed 1.0, default true
        void set_scale_up(const bool is_scale_up) {
            is_scale_up_ = is_scale_up;
        }

        /// Get is_scale_up, default true
        [[nodiscard]] bool get_scale_up() const { return is_scale_up_; }

        /// Set is_mini_pad, pad to the minimum rectange
      /// which height and width is times of stride
        void set_mini_pad(const bool is_mini_pad) {
            is_mini_pad_ = is_mini_pad;
        }

        /// Get is_mini_pad, default false
        [[nodiscard]] bool get_mini_pad() const { return is_mini_pad_; }

        /// Set padding stride, only for mini_pad mode
        void set_stride(const int stride) {
            stride_ = stride;
        }

        /// Get padding stride, default 32
        [[nodiscard]] bool get_stride() const { return stride_; }

    protected:
        bool preprocess(cv::Mat* mat, MDTensor* output,
                        std::map<std::string, std::array<float, 2>>* im_info);

        void letter_box(cv::Mat* mat) const;

        // target size, tuple of (width, height), default size = {640, 640}
        std::vector<int> size_;

        // padding value, size should be the same as channels
        std::vector<float> padding_value_;

        // only pad to the minimum rectangle which height and width is times of stride
        bool is_mini_pad_;

        // while is_mini_pad = false and is_no_pad = true,
        // will resize the image to the set size
        bool is_no_pad_;

        // if is_scale_up is false, the input image only can be zoom out,
        // the maximum resize scale cannot exceed 1.0
        bool is_scale_up_;

        // padding stride, for is_mini_pad
        int stride_;

        // for offset the boxes by classes when using NMS
        float max_wh_;
    };
}
