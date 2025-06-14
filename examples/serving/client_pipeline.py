import pickle

import tritonclient.http as httpclient
import cv2
import json

# 1. 读取图像并转为 RGB 格式，添加 batch 维度
image = cv2.imread('test_detection0.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None]  # shape: [1, H, W, 3]

# 2. Triton Server 地址
server_addr = '192.168.1.5:8000'

# 3. 创建 client
client = httpclient.InferenceServerClient(url=server_addr)

# 4. 模型名称与版本号
model_name = 'pipeline'
model_version = '1'

# 5. 构造输入
inputs = []
infer_input = httpclient.InferInput(name='INPUT', shape=image.shape, datatype='UINT8')
infer_input.set_data_from_numpy(image)
inputs.append(infer_input)

# 6. 构造预期输出
outputs = [
    httpclient.InferRequestedOutput('detction_result'),
]

# 7. 执行推理
response = client.infer(
    model_name=model_name,
    model_version=model_version,
    inputs=inputs,
    outputs=outputs
)

# 8. 获取输出
output0 = response.as_numpy('detction_result')
for j in range(len(output0)):
    value = output0[j][0]
    value = json.loads(value)
    print(value)
