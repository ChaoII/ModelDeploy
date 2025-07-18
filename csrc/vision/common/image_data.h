//
// Created by aichao on 2025/7/18.
//

#pragma once


#include <memory>
#include <vector>
#include <cstdint>
#include <string>
#include "core/md_decl.h"


namespace modeldeploy {
    class ImageDataImpl; // 前置声明

    class MODELDEPLOY_CXX_EXPORT ImageData {
    public:
        ImageData();
        ImageData(int width, int height, int channels, int type);
        ImageData(const ImageData& other);
        ImageData& operator=(const ImageData& other);
        ~ImageData();

        [[nodiscard]] int width() const;
        [[nodiscard]] int height() const;
        [[nodiscard]] int channels() const;
        [[nodiscard]] int type() const;
        [[nodiscard]] size_t data_size() const;
        [[nodiscard]] const uint8_t* data() const;
        uint8_t* data();


        [[nodiscard]] ImageData clone() const;
        static ImageData from_mat(const void* mat); // mat为cv::Mat*
        void to_mat(void* mat) const; // mat为cv::Mat*
        static void images_to_mats(const std::vector<ImageData>& images, const std::vector<void*>& mats);
        static ImageData imread(const std::string& filename);
        [[nodiscard]] bool imwrite(const std::string& filename) const;
        void imshow(const std::string& win_name) const;

    private:
        std::shared_ptr<ImageDataImpl> impl_; // PIMPL隐藏实现
    };
}
