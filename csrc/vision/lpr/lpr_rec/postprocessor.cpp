//
// Created by aichao on 2025/6/10.
//

#include "csrc/utils/utils.h"
#include "csrc/core/md_log.h"
#include "csrc/vision/lpr/lpr_rec/postprocessor.h"

namespace modeldeploy::vision::lpr {
    bool LprRecPostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<LprResult>* results) const {
        const size_t batch = tensors[0].shape()[0];
        results->reserve(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            if (tensors[0].dtype() != DataType::FP32 || tensors[1].dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            LprResult result;
            const float* plate_color_ptr = static_cast<const float*>(tensors[1].data()) + bs * tensors[1].shape()[1];
            const std::vector plate_color_vec(plate_color_ptr, plate_color_ptr + 5);
            int max_Index = argmax(plate_color_vec);
            const std::string plate_color = plate_color_list[max_Index];
            const size_t dim1 = tensors[0].shape()[1]; //21
            const size_t dim2 = tensors[0].shape()[2]; //78
            const float* prob1_temp_ptr = static_cast<const float*>(tensors[0].data()) + bs * dim1 * dim2;
            //车牌
            std::vector<int> plate_index;
            plate_index.reserve(dim1);
            for (size_t j = 0; j < dim1; j++) {
                std::vector plate_tensor(prob1_temp_ptr, prob1_temp_ptr + dim2);
                max_Index = argmax(plate_tensor);
                plate_index.push_back(max_Index);
                prob1_temp_ptr = prob1_temp_ptr + dim2;
            }
            int pre = 0;
            std::string plate_str;
            for (const int j : plate_index) {
                if (j != 0 && j != pre) {
                    plate_str += plate_chr[j];
                }
                pre = j;
            }
            result.car_plate_str = plate_str;
            result.car_plate_color = plate_color;
            results->push_back(std::move(result));
        }
        return true;
    }
}
