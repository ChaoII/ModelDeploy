# ModelDeploy 模型加密

ModelDeploy 使用 **AES-256-CBC** 实现模型加密，保护模型权重不被直接盗用。基于 OpenSSL / BCrypt（Windows 用 BCrypt，Linux 用 OpenSSL）。

> 编译需 `BUILD_ENCRYPTION=ON`（默认 ON）。未找到 OpenSSL 时加密功能静默禁用。

## 1. 加密文件格式

加密后的模型文件（`.mdenc`）结构：

```
[4 字节]  魔数 "MDEN" (ModelDeploy Encrypted)
[4 字节]  版本号（当前为 1）
[4 字节]  模型格式字符串长度
[N 字节]  模型格式字符串（如 "onnx", "mnn", "engine"）
[4 字节]  模型原始字节的 CRC32 校验和
[4 字节]  加密数据长度
[N 字节]  加密后的模型数据（AES-256-CBC + SHA-256 密钥派生）
```

## 2. 加密模型

使用命令行工具 `model_encrypted`：

```bash
# 加密
model_encrypted encrypt input_model_path output_model_path password [format]

# 示例：把 yolo11n.onnx 加密为 yolo11n_nms.mdenc
model_encrypted encrypt yolo11n.onnx yolo11n_nms.mdenc 123456 onnx

# 支持的格式：onnx / mnn / engine
```

工具源码：`examples/tools/model_encrypted.cpp`

## 3. 使用加密模型

加密模型的使用方式与未加密模型完全一致，只需在 `RuntimeOption` 中设置解密密码：

```cpp
modeldeploy::RuntimeOption option;
option.use_ort_backend();
option.use_cpu();
option.password = "123456";

auto det = modeldeploy::vision::detection::UltralyticsDet("yolo11n_nms.mdenc", option);
// 其余代码与普通模型完全相同
```

SDK 加载时自动识别 `.mdenc` 魔数，用密码解密后送入后端。

## 4. 解密模型

```bash
# 解密回原始格式
model_encrypted decrypt yolo11n_nms.mdenc yolo11n_decrypted.onnx 123456
```

## 5. 使用场景与注意事项

- **适用场景**：部署时防止模型权重被直接提取（如商业模型、隐私数据）
- **密码管理**：密码由部署方保管，写入代码或配置文件；被反编译代码也可能泄露，适合"防君子不防小人"的级别
- **性能影响**：仅模型**加载时**解密一次，推理性能无影响
- **格式识别**：加密文件内嵌模型格式，SDK 自动选择后端，无需额外指定

## 6. 与后端结合

加密模型适用于所有后端（ORT/TRT/MNN/Sophgo）：

```cpp
// TRT
option.use_trt_backend();
option.password = "123456";
auto det = UltralyticsDet("model.engine.mdenc", option);

// Sophgo
option.use_sophgo_backend(0);
option.password = "123456";
auto det = UltralyticsDet("model.bmodel.mdenc", option);
```
