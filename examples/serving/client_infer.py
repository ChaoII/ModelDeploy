import pickle
from concurrent.futures import ThreadPoolExecutor

import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def send_request(idx):
    # 1. 创建 client
    client = httpclient.InferenceServerClient(url="172.168.1.112:8000")
    # 2. 构造输入
    output0 = pickle.load(open("output0.pkl", "rb"))  # shape = [1, 3, H, W]
    print(f"[Client {idx}] input0 shape: {output0.shape}")
    inputs = []
    infer_input0 = httpclient.InferInput(name='images', shape=output0.shape, datatype=np_to_triton_dtype(output0.dtype))
    infer_input0.set_data_from_numpy(output0)
    inputs.append(infer_input0)
    # 3. 构造预期输出
    outputs = [
        httpclient.InferRequestedOutput('output0'),
    ]
    # 4. 执行推理
    response = client.infer(model_name="yolo11n", model_version="1", inputs=inputs, outputs=outputs)
    # 5. 获取输出
    output0 = response.as_numpy('output0')  # shape: [1, 6400, 84]
    pickle.dump(output0, open('infer_output0.pkl', 'wb'))
    print(f"[Client {idx}] Response OK, output0 shape: {output0.shape}")


with ThreadPoolExecutor(max_workers=7) as pool:
    pool.map(send_request, range(7))
