import pickle
from tritonclient.utils import np_to_triton_dtype

import tritonclient.http as httpclient
import numpy as np

output0 = pickle.load(open("output0.pkl", "rb"))

out0 = np.array(output0)

# 2. Triton Server 地址
server_addr = 'localhost:8000'

# 3. 创建 client
client = httpclient.InferenceServerClient(url=server_addr)

# 4. 模型名称与版本号
model_name = 'yolo11n'
model_version = '1'

# 5. 构造输入
inputs = []
infer_input0 = httpclient.InferInput(name='images', shape=out0.shape, datatype=np_to_triton_dtype(out0.dtype))
infer_input0.set_data_from_numpy(out0)
inputs.append(infer_input0)


# 6. 构造预期输出
outputs = [
    httpclient.InferRequestedOutput('output0'),
]

# 7. 执行推理
response = client.infer(
    model_name=model_name,
    model_version=model_version,
    inputs=inputs,
    outputs=outputs
)

# 8. 获取输出
output0 = response.as_numpy('output0')  # shape: [1, 3, H, W]

pickle.dump(output0, open('infer_output0.pkl', 'wb'))

