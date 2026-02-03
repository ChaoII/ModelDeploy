//
// Created by aichao on 2026/1/23.
//

#include "csrc/vision.h"

int main() {
    auto image1 = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_face_as_second.jpg");
    auto image2 = image1;
    auto image4 = image1.clone();
    auto image3 = std::move(image1);
    image2.rotate(ROTATE_90);
    image1.imshow("image1");
    image2.imshow("image2");
    image3.imshow("image3");
    image4.imshow("image4");
}
