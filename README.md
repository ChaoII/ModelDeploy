#### 1.编译

```bash
# 拉取源码
git clone https://github.com/ChaoII/ModelDeploy.git
# 配置生成 如果是msvc 请打开x64 Native Tools Command Prompt for VS 2022 终端 为了加速编译最好使用Ninja生成器
cmake -S . -B build -G Ninja -DBUILD_AUDIO=ON -DBUILD_VISION=ON -DBUILD_CAPI=OFF -DBUILD_PYTHON=OFF -DENABLE_MNN=OFF -DENABLE_ORT=ON -DENABLE_TRT=OFF -DWITH_GPU=OFF -DCMAKE_INSTALL_PREFIX=install
# 编译
cmake --build build --config Release --parallel
# 安装
cmake --install build
```

#### 2.模型加密

ModelDeploy采用XOR实现了简单的模型加密功能
##### 2.1 模型加密文件格式：
- [4字节] 魔数 "MDEN (ModelDeploy ENcrypted)
- [4字节] 版本号 (当前为1
- [4字节] 模型格式字符串长度
- [N字节] 模型格式字符串 (如 "onnx",mnn", engine)
- [4字节] 模型原始字节的CRC32校验和
- [4字节] 加密数据长度
- [N字节] 加密后的模型数据(XOR)
##### 2.2 模型加密方法：

```bash
model_encrypted encrypt input_model_path output_model_path password [format(mnn onnx engine)]
# 例如：
# 写入模型格式，ModelDeploy可以读取加密后的模型自动选择推理后端
model_encrypted encrypt yolo11n.onnx yolo11n_nms.mdenc 123456 onnx
```

##### 2.3 加密模型的使用
加密模型的使用与未加密模型的使用方式基本一致，在RuntimeOption中设置秘钥即可
```c++
modeldeploy::RuntimeOption option;
option.password = "123456";
modeldeploy::vision::detection::UltralyticsDet yolo11_det("yolo11n.mdenc", option);
...
```

**注意：**

1. `msvc`项目默认为`MD`版本，当使用`MT`版本静态库时，需要在`CMakeLists.txt`中修改如下：
2. 安装完成后会自动将ModelDeploySDK的后端依赖拷贝到lib目录中，比如开启`ort`，`mnn` 就会将`MNN.dll`和`onnxruntime.dll`
   拷贝进lib目录下
3. `ModelDeploySDK`需要完整的`c++17`标准支持的编译器进行编译
4. 对于使用`onnxruntime`后端`GPU`时，`windows`系统需要`windows10`、`windows11`、`windows server 2022` 及
   `windows server 2025`，其它系统请自行测试
5. `ModelDeploySDK`内置的`opencv`和`onnxruntime`静态库依赖是在`visual studio 2022` 和 `ubuntu 24.04`上编译，如果出现错误，请自行编译依赖
6. `ModelDeploySDK` 全部是基于`64`位系统来的，`2025`年了不要再用`32`位系统了

#### 2.使用方法(以windows为例)

编译并安装完成后会生成一个目录`install`, 包含一个`include`和`lib`子目录，其中`include`目录包含头文件，`lib`目录包含动态库和符号文件

- 创建一个新cmake项目`test_modeldeploy`,`CMakeLists.txt`文件中添加modeldeploy的头文件路径和库文件路径

```cmake
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(test_modeldeploy C CXX)
# 注意modeldeploy需要完整的c++17标准支持
set(CMAKE_CXX_STANDARD 17)
# msvc中必须添加该编译选项
if (MSVC)
    add_compile_options(/utf-8)
endif ()
# 设置modeldeploy的头文件目录和库文件牡蛎
set(MD_DIR "E:/CLionProjects/ModelDeploy/build/install")
set(MD_INC_DIR "${MD_DIR}/include")
set(MD_LIB_DIR "${MD_DIR}/lib")
include_directories(${MD_INC_DIR})
link_directories(${MD_LIB_DIR})
# 设置opencv目录（注意本SDK未将opencv显示导出，需要自行指定本地的opencv目录）
set(OpenCV_DIR "E:/develop/opencv5.x/x64/vc17/staticlib")
find_package(OpenCV REQUIRED)
# 添加可执行文件和依赖的库
add_executable(test_modeldeploy ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(test_modeldeploy ModelDeploySDK ${OpenCV_LIBS})
```

- 创建一个main.cpp文件，并添加以下代码

```c++
#include "modeldeploy/vision.h"
#include <opencv2/opencv.hpp>

int main() {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(10);
    option.use_gpu();
    option.use_trt_backend();
    option.use_gpu(0);
    option.enable_fp16 = true;
    option.enable_trt = true;
    option.ort_option.trt_engine_cache_path = "./trt_engine";
    // 注意：trt后端需要提前准备trtexec生成的.engine文件，下方有讲onnx转trt engine的命令
    modeldeploy::vision::detection::UltralyticsDet yolo11_det("./yolo11n_nms_dyn.engine",option);
    auto img = cv::imread("./test_person.jpg");
    std::vector<modeldeploy::vision::DetectionResult> result;
    yolo11_det.get_preprocessor().set_size({320, 320});
    int warming_up_count = 10;
    for (int i = 0; i < warming_up_count; ++i) {
        yolo11_det.predict(img, &result);
    }
    // 性能测试
    TimerArray timers;
    int loop_count = 100;
    for (int i = 0; i < loop_count; ++i) {
        yolo11_det.predict(img, &result, &timers);
    }
    timers.print_benchmark();
    const auto vis_image =
        modeldeploy::vision::vis_det(img, result, 0.3, "../../test_data/msyh.ttc", 12, 0.3,true);
    cv::imshow("test", vis_image);
    cv::waitKey(0);
}
```

更多示例请查看[example](./examples)

#### 3.OnnxRuntime使用混合精度推理

将fp32模型转换为fp16模型，在输入输出插入cast算子，将fp32转换为fp16，然后将输出参数从fp16转化为fp32

```python
import onnx
from onnxconverter_common import float16

# 加载原始 FP32 模型
model = onnx.load("model_fp32.onnx")
# 转为混合精度：内部节点为 float16，但输入/输出保持 float32
model_mixed = float16.convert_float_to_float16(
    model,
    keep_io_types=True
)
# 保存新模型
onnx.save(model_mixed, "model_mixed.onnx")
```

#### 4.OnnxRRuntime模型量化减小体积

此处仅为减小模型体积，使用uint8动态量化

```python
from onnxruntime.quantization import QuantType, quantize_dynamic

# 模型路径
model_fp32 = 'model_fp32.onnx'
model_quant_dynamic = 'model_quant_dynamic.onnx'

# 动态量化
quantize_dynamic(
    model_input=model_fp32,  # 输入模型
    # op_types_to_quantize=["Conv"],
    reduce_range=True,
    model_output=model_quant_dynamic,  # 输出模型
    per_channel=True,
    weight_type=QuantType.QUInt8,  # 参数类型 Int8 / UInt8
)
```

#### 5.trt engine生成

```bash
trtexec --onnx=yolo11n_nms.onnx ^
        --saveEngine=yolo11n_nms_dyn.engine ^
        --fp16 ^
        --minShapes=images:1x3x320x320 ^
        --optShapes=images:1x3x640x640 ^
        --maxShapes=images:4x3x1280x1280
```
