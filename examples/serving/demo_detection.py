import modeldeploy
import cv2
from pathlib import Path

base_path = Path("E:/CLionProjects/ModelDeploy/test_data")
image = cv2.imread(base_path / "test_images/test_detection0.jpg")
option = modeldeploy.RuntimeOption()
option.use_gpu()
option.enable_trt = True

model = modeldeploy.vision.UltralyticsDet(str(base_path / "test_models" / "yolo11n.onnx"), option)
# results = model.predict(image)
results = model.batch_predict([image])

# list[list[DetectionResult]]
print(results)
