from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tritonclient.http as httpclient
import cv2


def send_request(idx):
    client = httpclient.InferenceServerClient("192.168.1.5:8000")
    image = cv2.imread(f"test_detection{idx % 2}.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # shape: [1, H, W, 3]
    images = np.array([image])

    print(images.shape)

    infer_input = httpclient.InferInput("INPUT_0", images.shape, "UINT8")
    infer_input.set_data_from_numpy(images)

    outputs = [
        httpclient.InferRequestedOutput('preprocess_output_0'),
        httpclient.InferRequestedOutput('preprocess_output_1')
    ]

    response = client.infer("preprocess", inputs=[infer_input], outputs=outputs)
    print(f"[Client {idx}] Response OK, shape: {response.as_numpy('preprocess_output_0').shape}")


# send_request(0)
# # 使用 8 个线程并发请求
with ThreadPoolExecutor(max_workers=7) as pool:
    pool.map(send_request, range(7))

# # 8. 获取输出
# output0 = response.as_numpy('preprocess_output_0')  # shape: [1, 3, H, W]
# output1 = response.as_numpy('preprocess_output_1')  # shape: [1, ?] 字节格式
#
# pickle.dump(output0, open('output0.pkl', 'wb'))
# pickle.dump(output1, open('output1.pkl', 'wb'))
#
# # 9. 打印输出信息
# print("Output 0 shape:", output0.shape, output0.dtype)
# print("Output 1 (filename bytes):", output1)

