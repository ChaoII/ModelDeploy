//
// Created by aichao on 2026/1/19.
//

#include <opencv2/opencv.hpp>
#include "vision/common/convert.h"

namespace modeldeploy::vision {
    int md_image_type_to_ocv_type(const MdImageType type) {
        switch (type) {
        case MdImageType::GRAY_U8:
            return CV_8UC1;
        case MdImageType::GRAY_U16:
            return CV_16UC1;
        case MdImageType::GRAY_S16:
            return CV_16SC1;
        case MdImageType::GRAY_S32:
            return CV_32SC1;
        case MdImageType::GRAY_F32:
            return CV_32FC1;
        case MdImageType::GRAY_F64:
            return CV_64FC1;
        case MdImageType::PKG_BGR_U8:
        case MdImageType::PKG_RGB_U8:
            return CV_8UC3;
        case MdImageType::PKG_BGRA_U8:
        case MdImageType::PKG_RGBA_U8:
            return CV_8UC4;
        case MdImageType::PKG_BGR_F32:
        case MdImageType::PKG_RGB_F32:
            return CV_32FC3;
        case MdImageType::PKG_BGRA_F32:
        case MdImageType::PKG_RGBA_F32:
            return CV_32FC4;
        case MdImageType::PKG_BGR_F64:
        case MdImageType::PKG_RGB_F64:
            return CV_64FC3;
        case MdImageType::PKG_BGRA_F64:
        case MdImageType::PKG_RGBA_F64:
            return CV_64FC4;
        default:
            return -1;
        }
    }

    MdImageType md_image_type_from_ocv_type(const int ocv_type) {
        switch (ocv_type) {
        case CV_8UC1:
            return MdImageType::GRAY_U8;
        case CV_16UC1:
            return MdImageType::GRAY_U16;
        case CV_16SC1:
            return MdImageType::GRAY_S16;
        case CV_32SC1:
            return MdImageType::GRAY_S32;
        case CV_32FC1:
            return MdImageType::GRAY_F32;
        case CV_64FC1:
            return MdImageType::GRAY_F64;
        case CV_8UC3:
            return MdImageType::PKG_BGR_U8;
        case CV_8UC4:
            return MdImageType::PKG_BGRA_U8;
        case CV_32FC3:
            return MdImageType::PKG_BGR_F32;
        case CV_32FC4:
            return MdImageType::PKG_BGRA_F32;
        case CV_64FC3:
            return MdImageType::PKG_BGR_F64;
        case CV_64FC4:
            return MdImageType::PKG_BGRA_F64;
        default:
            return MdImageType::UNKNOWN;
        }
    }

    int md_color_convert_type_to_ocv_color_convert_type(const ColorConvertType md_type) {
        switch (md_type) {
        // 灰度转换
        case ColorConvertType::CVT_PA_BGR2GRAY:
            return cv::COLOR_BGR2GRAY;
        case ColorConvertType::CVT_PA_RGB2GRAY:
            return cv::COLOR_RGB2GRAY;

        // RGB/BGR 互转
        case ColorConvertType::CVT_PA_BGR2PA_RGB:
            return cv::COLOR_BGR2RGB;
        case ColorConvertType::CVT_PA_RGB2PA_BGR:
            return cv::COLOR_RGB2BGR;

        // 添加/移除 Alpha 通道
        case ColorConvertType::CVT_PA_BGR2PA_BGRA:
            return cv::COLOR_BGR2BGRA;
        case ColorConvertType::CVT_PA_RGB2PA_RGBA:
            return cv::COLOR_RGB2RGBA;
        case ColorConvertType::CVT_PA_BGR2PA_RGBA:
            return cv::COLOR_BGR2RGBA;
        case ColorConvertType::CVT_PA_RGB2PA_BGRA:
            return cv::COLOR_RGB2BGRA;
        case ColorConvertType::CVT_PA_BGRA2PA_BGR:
            return cv::COLOR_BGRA2BGR;
        case ColorConvertType::CVT_PA_RGBA2PA_RGB:
            return cv::COLOR_RGBA2RGB;
        case ColorConvertType::CVT_PA_RGBA2PA_BGR:
            return cv::COLOR_RGBA2BGR;
        case ColorConvertType::CVT_PA_BGRA2PA_RGB:
            return cv::COLOR_BGRA2RGB;
        case ColorConvertType::CVT_PA_BGRA2PA_RGBA:
            return cv::COLOR_BGRA2RGBA;
        case ColorConvertType::CVT_PA_RGBA2PA_BGRA:
            return cv::COLOR_RGBA2BGRA;

        // 灰度到彩色
        case ColorConvertType::CVT_GRAY2PA_RGB:
            return cv::COLOR_GRAY2RGB;
        case ColorConvertType::CVT_GRAY2PA_BGR:
            return cv::COLOR_GRAY2BGR;
        case ColorConvertType::CVT_GRAY2PA_BGRA:
            return cv::COLOR_GRAY2BGRA;
        case ColorConvertType::CVT_GRAY2PA_RGBA:
            return cv::COLOR_GRAY2RGBA;

        case ColorConvertType::CVT_PA_BGR2PAI420:
            return cv::COLOR_BGR2YUV_I420;
        case ColorConvertType::CVT_PA_RGB2PAI420:
            return cv::COLOR_RGB2YUV_I420;
        case ColorConvertType::CVT_PA_BGRA2PAI420:
            return cv::COLOR_BGRA2YUV_I420;
        case ColorConvertType::CVT_PA_RGBA2PAI420:
            return cv::COLOR_RGBA2YUV_I420;

        case ColorConvertType::CVT_NV122GRAY:
            return cv::COLOR_YUV2GRAY_NV12;
        case ColorConvertType::CVT_NV212GRAY:
            return cv::COLOR_YUV2GRAY_NV21;
        case ColorConvertType::CVT_I4202GRAY:
            return cv::COLOR_YUV2GRAY_I420;

        // NV12/NV21 到彩色
        case ColorConvertType::CVT_NV122PA_RGB:
            return cv::COLOR_YUV2RGB_NV12;
        case ColorConvertType::CVT_NV212PA_RGB:
            return cv::COLOR_YUV2RGB_NV21;
        case ColorConvertType::CVT_NV122PA_BGR:
            return cv::COLOR_YUV2BGR_NV12;
        case ColorConvertType::CVT_NV212PA_BGR:
            return cv::COLOR_YUV2BGR_NV21;
        case ColorConvertType::CVT_I4202PA_BGR:
            return cv::COLOR_YUV2BGR_I420;
        case ColorConvertType::CVT_I4202PA_RGB:
            return cv::COLOR_YUV2RGB_I420;

        // NV12/NV21 到带 Alpha
        case ColorConvertType::CVT_NV122PA_BGRA:
            return cv::COLOR_YUV2BGRA_NV12;
        case ColorConvertType::CVT_NV212PA_BGRA:
            return cv::COLOR_YUV2BGRA_NV21;
        case ColorConvertType::CVT_NV122PA_RGBA:
            return cv::COLOR_YUV2RGBA_NV12;
        case ColorConvertType::CVT_NV212PA_RGBA:
            return cv::COLOR_YUV2RGBA_NV21;
        case ColorConvertType::CVT_I4202PA_BGRA:
            return cv::COLOR_YUV2BGRA_I420;
        case ColorConvertType::CVT_I4202PA_RGBA:
            return cv::COLOR_YUV2RGBA_I420;

        default:
            return -1; // 不支持或无效转换
        }
    }
}
