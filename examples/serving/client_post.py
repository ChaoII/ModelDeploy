import json
import pickle
from concurrent.futures import ThreadPoolExecutor

import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def send_request(idx):
    # 1. 创建 client
    client = httpclient.InferenceServerClient(url="172.168.1.112:8000")

    # 2. 构造输入
    output0 = pickle.load(open("infer_output0.pkl", "rb"))  # shape = [1,84,6300]
    output1 = pickle.load(open("output1.pkl", "rb"))  # shape = [1,1]

    infer_input0 = httpclient.InferInput(name='POST_INPUT_0', shape=output0.shape,
                                         datatype=np_to_triton_dtype(output0.dtype))
    infer_input1 = httpclient.InferInput(name='POST_INPUT_1', shape=output1.shape,
                                         datatype=np_to_triton_dtype(output1.dtype))
    infer_input0.set_data_from_numpy(output0)
    infer_input1.set_data_from_numpy(output1)

    print(f"[Client {idx}] input0 shape: {output0.shape} | input1 shape: {output1.shape}")

    inputs = [
        infer_input0,
        infer_input1
    ]

    # 3. 构造预期输出
    outputs = [
        httpclient.InferRequestedOutput('POST_OUTPUT'),
    ]

    # 4. 执行推理
    response = client.infer(model_name="postprocess", model_version="1", inputs=inputs, outputs=outputs)
    # 5. 获取输出
    output0 = response.as_numpy('POST_OUTPUT')  # shape=[1,]
    value = json.loads(output0[0])
    print(f"[Client {idx}] Response OK, output0 shape: {output0.shape}, value is: {value}")


# # 使用 8 个线程并发请求
with ThreadPoolExecutor(max_workers=7) as pool:
    pool.map(send_request, range(7))
