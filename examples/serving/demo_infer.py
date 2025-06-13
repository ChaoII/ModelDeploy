import modeldeploy
import cv2
from pathlib import Path
import numpy as np

base_path = Path("E:/CLionProjects/ModelDeploy/test_data")
image = cv2.imread(base_path / "test_images/test_detection.jpg")
option = modeldeploy.RuntimeOption()
option.set_model_path(str(base_path / "test_models" / "yolo11n.onnx"))
image_ = cv2.imread("vis_result.jpg")

# 预处理是带batch的
preprocessor = modeldeploy.vision.UltralyticsPreprocessor()
(pre_images, letter_box_records) = preprocessor.run([image])

print(pre_images)

runtime = modeldeploy.Runtime()
runtime.init(option)
inputs = dict()
for i in range(runtime.num_inputs()):
    print(runtime.get_input_info(i))
    inputs.update({runtime.get_input_info(i).name: pre_images[i]})
run_time_result = runtime.infer(inputs)
postprocessor = modeldeploy.vision.UltralyticsPostprocessor()
# 包含batch维度
results = postprocessor.run(run_time_result, letter_box_records)

# batch
for result in results:
    # box
    for result_ in result:
        print(result_.box)
