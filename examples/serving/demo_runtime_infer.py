from pathlib import Path

import cv2
import modeldeploy

base_path = Path("E:/CLionProjects/ModelDeploy/test_data")
image = cv2.imread(base_path / "test_images/test_detection0.jpg")
option = modeldeploy.RuntimeOption()
option.set_model_path(str(base_path / "test_models" / "yolo11n.onnx"))

# 预处理是带batch的
preprocessor = modeldeploy.vision.UltralyticsPreprocessor()
# list[Tensor] List[LetterBoxRecord] list[np.array]
(pre_images, letter_box_records) = preprocessor.run([image])

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
print(results)
