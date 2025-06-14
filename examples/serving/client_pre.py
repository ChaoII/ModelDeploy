import pickle

import tritonclient.http as httpclient
import cv2

# 1. 读取图像并转为 RGB 格式，添加 batch 维度
image = cv2.imread('test_detection0.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None]  # shape: [1, H, W, 3]

# 2. Triton Server 地址
server_addr = 'localhost:8000'

# 3. 创建 client
client = httpclient.InferenceServerClient(url=server_addr)

# 4. 模型名称与版本号
model_name = 'preprocess'
model_version = '1'

# 5. 构造输入
inputs = []
infer_input = httpclient.InferInput(name='INPUT_0', shape=image.shape, datatype='UINT8')
infer_input.set_data_from_numpy(image)
inputs.append(infer_input)

# 6. 构造预期输出
outputs = [
    httpclient.InferRequestedOutput('preprocess_output_0'),
    httpclient.InferRequestedOutput('preprocess_output_1')
]

# 7. 执行推理
response = client.infer(
    model_name=model_name,
    model_version=model_version,
    inputs=inputs,
    outputs=outputs
)

# 8. 获取输出
output0 = response.as_numpy('preprocess_output_0')  # shape: [1, 3, H, W]
output1 = response.as_numpy('preprocess_output_1')  # shape: [1, ?] 字节格式

pickle.dump(output0, open('output0.pkl', 'wb'))
pickle.dump(output1, open('output1.pkl', 'wb'))

# 9. 打印输出信息
print("Output 0 shape:", output0.shape, output0.dtype)
print("Output 1 (filename bytes):", output1)
