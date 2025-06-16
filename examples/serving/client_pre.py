import pickle
from concurrent.futures import ThreadPoolExecutor

import cv2
import tritonclient.http as httpclient


def send_request(idx):
    # 1. 创建 client
    client = httpclient.InferenceServerClient("172.168.1.112:8000")
    # 2. 构造输入
    image = cv2.imread(f"test_detection{idx % 2}.jpg")
    images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None]  # shape: [1, H, W, 3]
    print(f"[Client {idx}] input0 shape: {images.shape}")

    infer_input = httpclient.InferInput("INPUT_0", images.shape, "UINT8")
    infer_input.set_data_from_numpy(images)
    inputs = [
        infer_input
    ]
    # 3. 构造预期输出
    outputs = [
        httpclient.InferRequestedOutput('preprocess_output_0'),
        httpclient.InferRequestedOutput('preprocess_output_1')
    ]
    # 4. 执行推理
    response = client.infer("preprocess", inputs=inputs, outputs=outputs)
    # 5. 获取输出
    output0 = response.as_numpy('preprocess_output_0')  # shape: [1, 3, H, W]
    output1 = response.as_numpy('preprocess_output_1')  # shape: [1, 1] 字节格式
    pickle.dump(output0, open('output0.pkl', 'wb'))
    pickle.dump(output1, open('output1.pkl', 'wb'))
    print(f"[Client {idx}] Response OK, output0 shape: {output0.shape}|output1 shape: {output1.shape} ")


# send_request(0)
# # 使用 8 个线程并发请求
with ThreadPoolExecutor(max_workers=7) as pool:
    pool.map(send_request, range(7))
