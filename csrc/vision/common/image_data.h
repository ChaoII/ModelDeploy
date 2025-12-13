//
// Created by aichao on 2025/7/18.
//

#pragma once


#include <memory>
#include <vector>
#include <cstdint>
#include <string>
#include "core/md_decl.h"
#include "core/tensor.h"
#include "vision/common/struct.h"

namespace cv {
    class Mat;
}

namespace modeldeploy {
    class ImageDataImpl; // 前置声明


    class MODELDEPLOY_CXX_EXPORT ImageData {
    public:
        ImageData();
        ImageData(int width, int height, int channels, int type);
        ImageData(const ImageData& other);
        explicit ImageData(const cv::Mat* mat);
        ImageData& operator=(const ImageData& other);
        ~ImageData();

        [[nodiscard]] int width() const;
        [[nodiscard]] int height() const;
        [[nodiscard]] int channels() const;
        [[nodiscard]] int type() const;
        [[nodiscard]] size_t data_size() const;
        [[nodiscard]] const uint8_t* data() const;
        uint8_t* data();

        ImageData crop(const vision::Rect2f& rect) const;

        void to_tensor(Tensor* tensor) const;

        void letter_box(const std::vector<int>& size,
                        const std::vector<float>& padding_value,
                        vision::LetterBoxRecord* letter_box_record) const;

        void convert_and_permute(const std::vector<float>& alpha, const std::vector<float>& beta,
                                 bool swap_rb = true) const;

        [[nodiscard]] bool empty() const;

        [[nodiscard]] ImageData clone() const;
        static ImageData from_raw(const unsigned char* data, int width, int height, int channels);
        static ImageData from_raw_nocopy(unsigned char* data, int width, int height, int channels);
        static ImageData from_mat(const cv::Mat* mat_ptr); // mat为cv::Mat*
        void update_from_mat(const cv::Mat* mat_ptr, bool is_copy = false);
        void to_mat(cv::Mat* mat_ptr, bool is_copy = false) const; // mat为cv::Mat*
        static void images_to_mats(const std::vector<ImageData>& images, const std::vector<cv::Mat*>& mats);
        static ImageData imread(const std::string& filename);
        [[nodiscard]] bool imwrite(const std::string& filename) const;
        void imshow(const std::string& win_name) const;

    private:
        std::shared_ptr<ImageDataImpl> impl_; // PIMPL隐藏实现
    };
}
