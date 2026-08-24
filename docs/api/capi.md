# C API（`md_*` 前缀）

面向 C/C++ 嵌入式、其他语言 FFI 桥接。统一返回 `MDStatusCode`。

```c
#include "modeldeploy/md_model_capi.h"

// 创建模型
MDModel model = md_create_detection_model("yolo11n.onnx", md_create_default_runtime_option());

// 设置输入尺寸
md_set_detection_input_size(model, 640, 640);

// 推理
MDDetectionResults results;
md_detection_predict(model, img, &results);

// 读取结果
int n = md_get_detection_result_size(results);
// ...

// 释放
md_free_detection_results(results);
md_free_detection_model(model);
```

## 接口分组

| 模块 | 接口 |
|------|------|
| 检测 | `md_create_detection_model` / `md_detection_predict` |
| 分类 | `md_create_classification_model` / `md_classification_predict` |
| 分割 | `md_create_instance_seg_model` / `md_instance_seg_predict` |
| 姿态 | `md_create_keypoint_model` / `md_keypoint_predict` |
| 旋转框 | `md_create_obb_model` / `md_obb_predict` |
| 人脸 | `md_create_face_det/rec/age/gender/as_*_model` |
| 车牌 | `md_create_lpr_*_model` |
| OCR | `md_create_ocr_model` / `md_ocr_model_predict` |
| 行人属性 | `md_create_attr_model` / `md_attr_predict` |
| 图像 | `md_read_image` / `md_save_image` / `md_from_bgr24` 等 |
| 绘制 | `md_draw_rect` / `md_draw_polygon` / `md_draw_text` |

编译需 `BUILD_CAPI=ON`。
