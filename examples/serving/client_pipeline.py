import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import tritonclient.http as httpclient


def send_request(idx):
    # 1. 创建 client
    client = httpclient.InferenceServerClient("172.168.1.112:8000")
    image = cv2.imread(f"test_detection{idx % 2}.jpg")
    images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None]  # shape: [1, H, W, 3]
    print(f"[Client {idx}] input0 shape: {images.shape}")
    # 2. 构造输入
    infer_input = httpclient.InferInput("INPUT", images.shape, "UINT8")
    infer_input.set_data_from_numpy(images)
    inputs = [
        infer_input
    ]
    # 3. 构造预期输出
    outputs = [
        httpclient.InferRequestedOutput('detection_result')
    ]
    # 4. 执行推理
    response = client.infer(model_name="pipeline", model_version="1", inputs=inputs, outputs=outputs)

    # 5. 获取输出
    output0 = response.as_numpy('detection_result')  # shape: [1,]
    value = json.loads(output0[0])
    print(f"[Client {idx}] Response OK, output0 shape: {output0.shape},value is: {value}")


# # 使用 8 个线程并发请求
with ThreadPoolExecutor(max_workers=7) as pool:
    pool.map(send_request, range(7))
