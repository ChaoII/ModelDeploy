//
// Created by aichao on 2025/2/8.
//

#pragma once

#include "capi/common/md_types.h"
#include "capi/common/md_decl.h"

#ifdef __cplusplus
extern "C" {
#endif
/// 弹窗显示image
/// \param image
MODELDEPLOY_CAPI_EXPORT void md_show_image(MDImage* image);


/// 根据对应位置的图像裁剪（注意需要手动释放MDImage）
/// \param image 原始图像
/// \param rect 待裁剪的区域
/// \return 裁剪后的图像
MODELDEPLOY_CAPI_EXPORT MDImage md_crop_image(MDImage* image, const MDRect* rect);


/// 克隆一个MDImage(注意需要手动释放WIMage指针)
/// \param image
/// \return MDImage 指针
MODELDEPLOY_CAPI_EXPORT MDImage md_clone_image(const MDImage* image);


/// 从压缩字节生成一个MDImage指针（需要手动释放）
/// \param bytes 压缩字节，比如.jpg的buffer数据
/// \param size 字节长度
/// \return MDImage指针
MODELDEPLOY_CAPI_EXPORT MDImage md_from_compressed_bytes(const unsigned char* bytes, int size);

/// 从原始的BGR24 数据生成一个MDImage对象（注意需要手动释放MDImage）。
///
/// @param data BGR24 数据。
/// @param width 图像宽度。
/// @param height 图像高度。
/// @return 返回一个MDImage对象，表示生成的图像。
MODELDEPLOY_CAPI_EXPORT MDImage md_from_bgr24_data(const unsigned char* data, int width, int height);

/// 从原始的RGB24 数据生成一个MDImage对象（注意需要手动释放MDImage）。
/// @param data RGB24 数据。
/// @param width 图像宽度。
/// @param height 图像高度。
/// @return 返回一个MDImage对象，表示生成的图像。
///
MODELDEPLOY_CAPI_EXPORT MDImage md_from_rgb24_data(const unsigned char* data, int width, int height);

///  从RGB24数据转换为BGR24数据
/// @param data rgb24数据
/// @param width 图像宽度。
/// @param height 图像高度。
/// @return MDImage
MODELDEPLOY_CAPI_EXPORT MDImage md_from_rgb24_data_to_bgr24(const unsigned char* data, int width, int height);



///  从base64编码的字符串生成一个MDImage对象（注意需要手动释放MDImage）。
///
///  @param base64_str base64编码的字符串，表示图像数据。
///  @return 返回一个MDImage对象，表示解码后的图像。
MODELDEPLOY_CAPI_EXPORT MDImage md_from_base64_str(const char* base64_str);

/// 从文件读取MDImage
/// \param path 图像路径
/// \return MDImage指针
MODELDEPLOY_CAPI_EXPORT MDImage md_read_image(const char* path);

/// 从指定设备id中获取图像
/// @param device_id 设备id比如默认摄像头为0
/// @param frame_width 帧宽
/// @param frame_height 帧高
/// @param is_save_file 是否保存文件
/// @return image
MODELDEPLOY_CAPI_EXPORT MDImage md_read_image_from_device(int device_id, int frame_width = 640, int frame_height = 480,
                                                          bool is_save_file = false);

/// 保存图片
/// @param image 图像
/// @param path 图片途径
MODELDEPLOY_CAPI_EXPORT void md_save_image(MDImage* image, const char* path);

/// 释放MDImage指针
/// \param image
MODELDEPLOY_CAPI_EXPORT void md_free_image(MDImage* image);

#ifdef __cplusplus
}
#endif
