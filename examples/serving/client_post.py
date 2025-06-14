import json
import pickle

import modeldeploy
from tritonclient.utils import np_to_triton_dtype

import tritonclient.http as httpclient
import numpy as np

output0 = pickle.load(open("infer_output0.pkl", "rb"))
output1 = pickle.load(open("output1.pkl", "rb"))

out0 = np.array(output0)
out1 = np.array(output1, dtype=np.object_)

# 2. Triton Server 地址
server_addr = '192.168.1.5:8000'

# 3. 创建 client
client = httpclient.InferenceServerClient(url=server_addr)

# 4. 模型名称与版本号
model_name = 'postprocess'
model_version = '1'

# 5. 构造输入
inputs = []
infer_input0 = httpclient.InferInput(name='POST_INPUT_0', shape=out0.shape, datatype=np_to_triton_dtype(out0.dtype))
infer_input1 = httpclient.InferInput(name='POST_INPUT_1', shape=out1.shape, datatype=np_to_triton_dtype(out1.dtype))
infer_input0.set_data_from_numpy(out0)
infer_input1.set_data_from_numpy(out1)
inputs.append(infer_input0)
inputs.append(infer_input1)

# 6. 构造预期输出
outputs = [
    httpclient.InferRequestedOutput('POST_OUTPUT'),
]

# 7. 执行推理
response = client.infer(
    model_name=model_name,
    model_version=model_version,
    inputs=inputs,
    outputs=outputs
)

# 8. 获取输出
output0 = response.as_numpy('POST_OUTPUT')
for out_ in output0:
    r = json.loads(out_.decode())
    print(r)
