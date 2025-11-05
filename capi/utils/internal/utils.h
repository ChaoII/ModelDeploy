//
// Created by AC on 2024-12-24.
//
#pragma once

#include "opencv2/opencv.hpp"


#include "csrc/vision/common/image_data.h"
#include "csrc/runtime/runtime_option.h"
#include "csrc/vision/common/result.h"
#include "capi/common/md_types.h"

///
/// 将MDImage对象转换为OpenCV的Mat对象
///
/// @param image 指向MDImage对象的指针，该对象包含了图像数据
/// @return 返回一个OpenCV的Mat对象，包含了从MDImage对象转换来的图像数据
///
/// 此函数的目的是为了在MDImage图像处理库和OpenCV之间建立一座桥梁
/// 通过将MDImage对象转换为OpenCV的Mat对象，使得用户可以利用OpenCV丰富的图像处理功能
/// 对MDImage对象进行操作此函数确保了不同图像处理库之间的兼容性，提高了代码的灵活性和可用性
///
modeldeploy::ImageData md_image_to_image_data(const MDImage* image);


cv::Mat md_image_to_mat(const MDImage* image);

///
/// 将OpenCV的Mat对象转换为MDImage对象。
///
/// @param mat 输入的OpenCV Mat对象，包含图像数据。
/// @return 返回一个MDImage对象，表示转换后的图像。
///
/// 此函数的目的是在OpenCV图像处理库和MDImage图像表示之间提供一个接口，
/// 允许在不同图像处理场景下互操作。通过封装Mat对象到MDImage中，可以更容易地
/// 在基于MDImage的框架或应用程序中利用OpenCV的功能。
///
MDImage* image_data_to_md_image(const modeldeploy::ImageData& mat);

MDImage* mat_to_md_image(const cv::Mat& mat);

#ifdef BUILD_FACE

///
/// 将MDImage类型的图像转换为SeetaImageData类型的图像
///
/// @param image 指向MDImage类型图像的指针，这是输入图像数据
/// @return 返回一个SeetaImageData类型的图像，这是转换后的图像数据
///
/// 此函数的目的是为了在不同图像处理库之间进行图像数据转换，以便在使用不同库的函数时保持兼容性
///
SeetaImageData md_image_to_seeta_image(const MDImage* image);

#endif

///
/// 在图像上绘制半透明矩形的内部函数。
///
/// 该函数接收一个图像、一个矩形区域、一种颜色和一个透明度参数，然后在指定的图像上绘制出半透明的矩形。
/// 主要用于在图像处理过程中标示出特定区域，使得该区域以半透明的方式突出显示，以便于分析或调试。
///
/// @param cv_image 需要绘制矩形的图像，类型为cv::Mat。这个图像是OpenCV框架下的一种数据结构，用于表示图像数据。
/// @param rect 要绘制的矩形区域，类型为cv::Rect。它包括矩形的左上角坐标、宽度和高度，用于定义矩形的位置和大小。
/// @param cv_color 矩形的颜色，类型为cv::Scalar。在OpenCV中，cv::Scalar用于表示颜色，通常包含BGR（蓝、绿、红）三个分量。
/// @param alpha 透明度参数，类型为double。其值范围在0.0到1.0之间，0.0表示完全透明，1.0表示完全不透明。
///              这个参数决定了绘制出的矩形的透明程度。
///
/// 注：此函数名称以_internal结尾，表明这是一个内部实现函数，可能不直接供外部调用，而是被其他函数调用以完成特定的绘制任务。
void draw_rect_internal(cv::Mat& cv_image, const cv::Rect& rect, const cv::Scalar& cv_color, double alpha);

///  在图像中绘制多边形的内部。
///
///  此函数的目的是在给定的图像中填充一个多边形的内部区域。它使用了透明度概念来混合多边形的颜色和原有图像的颜色，
///  从而实现平滑的视觉效果。该函数不处理多边形的轮廓绘制，仅关注内部区域的填充。
///
///  @param cv_image 需要绘制多边形的图像。这是一个引用参数，意味着对它的任何修改都会反映在原始图像上。
///  @param points 一个包含多边形顶点的向量。这些点的顺序决定了多边形的形状。
///  @param color 指定多边形填充颜色的标量。在OpenCV中，这通常是一个包含BGR（蓝、绿、红）值的数组。
///  @param alpha 透明度因子，其值在0到1之间。它决定了多边形颜色和原有图像颜色的混合程度。0表示完全透明（即不改变原始图像），1表示完全不透明（即完全覆盖原始图像）。
void draw_polygon_internal(cv::Mat& cv_image, const std::vector<cv::Point>& points,
                           const cv::Scalar& color, double alpha);


/// 在图像上绘制文本的内部函数。
///
/// 该函数负责在给定的图像和矩形区域内绘制指定的文本字符串。它可以使用指定的字体路径和大小来定制文本的外观，
/// 并且可以通过颜色和透明度参数来定制文本的渲染效果。
///
/// @param cv_image 待绘制文本的图像。
/// @param rect 定义文本绘制区域的矩形。
/// @param text 要绘制的文本字符串。
/// @param font_path 字体文件的路径，用于文本的样式。
/// @param font_size 文本的字体大小。
/// @param cv_color 文本的颜色，使用OpenCV的Scalar类型表示。
/// @param alpha 文本的透明度，范围为0.0到1.0，0.0表示完全透明，1.0表示完全不透明。
void draw_text_internal(cv::Mat& cv_image, const cv::Rect& rect, const std::string& text, const std::string& font_path,
                        int font_size, const cv::Scalar& cv_color, double alpha);


/// 在图像上绘制文本的内部函数。
///
/// @param cv_image 需要绘制文本的图像。
/// @param point 文本的起始位置。
/// @param text 要绘制的文本内容。
/// @param font_path 字体文件的路径。
/// @param font_size 文本的字体大小。
/// @param cv_color 文本的颜色。
void draw_text_internal(cv::Mat& cv_image, const cv::Point& point, const std::string& text,
                        const std::string& font_path, int font_size, const cv::Scalar& cv_color);


/// 检查字符串中是否包含子字符串
///
/// @param str 被检查的源字符串
/// @param sub_str 需要查找的子字符串
/// @return 如果源字符串中包含子字符串，则返回true；否则返回false
bool contains_substring(const std::string& str, const std::string& sub_str);


///
/// 将多边形格式化为字符串
/// 用于将MDPolygon对象转换为描述多边形的字符串格式
/// 主要用于调试和日志记录
///
/// @param polygon MDPolygon对象，代表一个多边形
/// @return 描述多边形的字符串
std::string format_polygon(MDPolygon polygon);

///
/// 将矩形格式化为字符串
/// 用于将MDRect对象转换为描述矩形的字符串格式
/// 主要用于调试和日志记录
///
/// @param rect MDRect对象，代表一个矩形
/// @return 描述矩形的字符串
std::string format_rect(MDRect rect);


///
/// 从源图像中获取旋转裁剪后的图像
/// 根据指定的多边形从源图像中裁剪出旋转的子图像，通常用于图像处理和特征提取
///
/// @param src_image 原始图像，从中进行裁剪
/// @param polygon 指向MDPolygon对象的指针，定义裁剪区域
/// @return 旋转裁剪后的图像
cv::Mat get_rotate_crop_image(const cv::Mat& src_image, const MDPolygon* polygon);


//---------------------------for result type--------------------------------

using namespace modeldeploy::vision;


void classification_result_2_c_results(
    const ClassifyResult& result,
    MDClassificationResults* c_results);


void c_results_2_classification_result(
    const MDClassificationResults* c_results,
    ClassifyResult* result);


void detection_results_2_c_results(
    const std::vector<DetectionResult>& results,
    MDDetectionResults* c_results);


void c_results_2_detection_results(
    const MDDetectionResults* c_results,
    std::vector<DetectionResult>* results);


void obb_results_2_c_results(
    const std::vector<ObbResult>& results,
    MDObbResults* c_results);


void c_results_2_obb_results(
    const MDObbResults* c_results,
    std::vector<ObbResult>* results);


void iseg_results_2_c_results(
    const std::vector<InstanceSegResult>& results,
    MDIsegResults* c_results);


void c_results_2_iseg_results(
    const MDIsegResults* c_results,
    std::vector<InstanceSegResult>* results);


void keypoint_results_2_c_results(
    const std::vector<KeyPointsResult>& results,
    MDKeyPointResults* c_results);


void c_results_2_keypoint_results(
    const MDKeyPointResults* c_results,
    std::vector<KeyPointsResult>* results);


void ocr_result_2_c_results(
    const OCRResult& result,
    MDOCRResults* c_results);


void c_results_2_ocr_result(
    const MDOCRResults* c_results,
    OCRResult* result);


void lpr_results_2_c_results(
    const std::vector<LprResult>& results,
    MDLPRResults* c_results);

void c_results_2_lpr_results(
    const MDLPRResults* c_results,
    std::vector<LprResult>* results);


void face_recognizer_result_2_c_result(
    const FaceRecognitionResult& result,
    MDFaceRecognizerResult* c_result);


void c_result_2_face_recognizer_result(
    const MDFaceRecognizerResult* c_result,
    FaceRecognitionResult* result);


void face_recognizer_results_2_c_results(
    const std::vector<FaceRecognitionResult>& results,
    MDFaceRecognizerResults* c_results);


void c_results_2_face_recognizer_results(
    const MDFaceRecognizerResults* c_results,
    std::vector<FaceRecognitionResult>* results);

void c_runtime_option_2_runtime_option(
    const MDRuntimeOption* c_option,
    modeldeploy::RuntimeOption* option);
