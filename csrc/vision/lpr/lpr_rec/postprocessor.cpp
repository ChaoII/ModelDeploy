//
// Created by aichao on 2025/6/10.
//

#include "utils/utils.h"
#include "core/md_log.h"
#include "vision/lpr/lpr_rec/postprocessor.h"

namespace modeldeploy::vision::lpr {
    bool LprRecPostprocessor::run(
        std::vector<Tensor>& tensors, std::vector<LprResult>* results) const {
        // ncnn batch==1 压掉首维，且双输出槽位与 ORT/ONNX 反序：
        // ncnn 实际 t0=color[5](1D)、t1=rec[21,78](2D)，而 ORT 为 t0=rec[1,21,78]、t1=color[1,5]。
        // 先按 ORT 槽位交换为（rec, color），再各自 expand_dim(0) 补 batch 维后落体。
        if (tensors.size() >= 2 && tensors[0].shape().size() == 1) {
            const Tensor color = tensors[0];
            tensors[0] = tensors[1];
            tensors[1] = color;
            tensors[0].expand_dim(0);
            tensors[1].expand_dim(0);
        }
        const size_t batch = tensors[0].shape()[0];
        results->resize(batch);
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
            results->at(bs) = std::move(result);
        }
        return true;
    }
}
