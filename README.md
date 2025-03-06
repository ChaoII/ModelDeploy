#### 1.编译

```bach
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
