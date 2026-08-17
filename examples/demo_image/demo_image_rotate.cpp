#include "csrc/vision.h"

int main() {
    auto original = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_face_as_second.jpg");
    auto rotated = original.clone();
    rotated.rotate(ROTATE_90);
    if (!original.imwrite("capi2_rotate_original.jpg")) return 1;
    if (!rotated.imwrite("capi2_rotate90_out.jpg")) return 1;
    return 0;
}
