//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <chrono>
#include "src/utils/utils_capi.h"
#include "src/ocr/ocr_capi.h"
#ifdef WIN32
#include <windows.h>
#endif
int main() {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    MDStatusCode ret;
    //��ȡͼƬ
    auto image = md_read_image("../tests/test_images/test_ocr1.png");
    // ��Ҫ���ҵ��ı�
    const char *text = "��ͣ����";
    // ����ģ�;��
    MDModel model;
    MDOCRModelParameters parameters = {
            "../tests/models/ocr",
            "../tests/key.txt",
            8,
            PaddlePaddle,
            960,
            0.3,
            0.6,
            1.5,
            "slow",
            0,
            4
    };
    if ((ret = md_create_ocr_model(&model, &parameters)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    // ��ȡ�ı�Ŀ��λ��
    MDOCRResults results;
    if ((ret = md_ocr_model_predict(&model, &image, &results)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    MDColor color = {0, 0, 255};
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    md_draw_ocr_result(&image, &results, "../tests/msyh.ttc", 15, &color, 0.5, 1);
    std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    std::cout << "cost: " << diff.count() << std::endl;
    md_print_ocr_result(&results);
    md_free_ocr_result(&results);
    auto rect = md_get_text_position(&model, &image, text);
    // ��ӡ�ı���Ϣ
    md_print_rect(&rect);
    // �Ȳü��ٻ��ƣ���Ȼimage��ָ�봫�ݣ��ڻ���ʱ���޸�ԭʼimage
    if (rect.width > 0 && rect.height > 0) {
        auto roi = md_crop_image(&image, &rect);
        // ��ԭʼ�����ϻ����ı����
        md_draw_text(&image, &rect, text, "../tests/msyh.ttc", 15, &color, 0.5);
        auto enable = md_get_button_enable_status(&roi, 50, 0.05);
        // �жϰ�ť�Ƿ����
        std::cout << "enable: " << enable << std::endl;
//        md_show_image(&roi);
        // �ͷ���Դ
        md_free_image(&roi);
    }

    // ��ʾԭʼ����
//    md_show_image(&image);
    // ��ʾĿ���ı����ڻ���
    md_free_image(&image);
    md_free_ocr_model(&model);
    return ret;

}