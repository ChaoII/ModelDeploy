#### 1.编译

1. `release/v1.0`的audio部分使用了funasr sdk源码，具有2pass功能，但是相对不灵活，缺失tts模型之后引用了sherpa-onnx源码
2. 从`release/v1.1` 开始人脸识别部分全部转换为onnxruntime推理，放弃了seetaface的tennis推理引擎 
3. 从`release/v1.2` 为老版本的`md_tensor.h`的方案，其优势是灵活可以与不同硬件设备直接集成，但是编译速度很慢很慢，之后重构了tensor类，编译速度大大提升，库体积大小与重构前相当 
4. 如果主干分支bug多，速度慢，请切换至`release/v1.2`分支

```bash
git clone https://github.com/ChaoII/ModelDeploy.git
cmake -S . -B build -DBUILD_AUDIO=ON -DBUILD_VISION=ON -DBUILD_FACE=ON -DBUILD_CAPI=ON
cmake --build build --config Release
```

经过以上命令后会生成一个`ModelDeploySDK`动态库
注意编译过程中的一些依赖在[百度网盘获取](https://pan.baidu.com/s/1HkutYev3GkKYTvn6NKmBFA?pwd=j5z7)

1. `OnnxRuntime`和`OpenCV`静态库均存在`MT`和`MD`版本，请配套使用，
2. 在开启`-DBUILD_AUDIO`选项后，依赖很多第三方库，第三方库的源码在`audio_deps`文件夹下，将里面的所有压缩包放进项目的`$
   {CMAKE_BINARY_DIR}`即可，会自动解压并依赖
3. `msvc`项目默认为`MD`版本，当使用`MT`版本静态库时，需要在`CMakeLists.txt`中修改如下：

```CMakeLists.txt
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MT")
```
