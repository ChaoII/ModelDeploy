//
// Created by aichao on 2025/2/8.
//

#pragma once

#include "csrc/common/md_types.h"
#include "csrc/common/md_decl.h"

#ifdef __cplusplus
extern "C" {
#endif
/// 弹窗显示image
/// \param image
EXPORT_DECL void md_show_image(MDImage* image);


/// 根据对应位置的图像裁剪（注意需要手动释放MDImage）
/// \param image 原始图像
/// \param rect 待裁剪的区域
/// \return 裁剪后的图像
EXPORT_DECL MDImage md_crop_image(MDImage* image, const MDRect* rect);


/// 克隆一个MDImage(注意需要手动释放WIMage指针)
/// \param image
/// \return MDImage 指针
EXPORT_DECL MDImage md_clone_image(MDImage* image);


/// 从压缩字节生成一个MDImage指针（需要手动释放）
/// \param bytes 压缩字节，比如.jpg的buffer数据
/// \param size 字节长度
/// \return MDImage指针
EXPORT_DECL MDImage md_from_compressed_bytes(const unsigned char* bytes, int size);


/// 从文件读取MDImage
/// \param path 图像路径
/// \return MDImage指针
EXPORT_DECL MDImage md_read_image(const char* path);

EXPORT_DECL MDImage md_read_image_from_device(int device_id, int frame_width = 640, int frame_height = 480,
                                              bool is_save_file = false);


/// 释放MDImage指针
/// \param img
EXPORT_DECL void md_free_image(MDImage* img);

#ifdef __cplusplus
}
#endif