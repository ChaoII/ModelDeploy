# bmodel 精度验证(cmodel vs ONNX, int8/qtable 混合精度/F16)

| key | 状态 | cos/max-err | 输出量纲 |
|---|---|---|---|
| yolo26n | OK | cos=0.9996 max=413.2 | onnx_max=656 bmodel_max=664 |
| yolo26n-cls | OK | cos=0.8199 max=0.1508 | onnx_max=0.229 bmodel_max=0.255 |
| yolo26n-obb | OK | cos=0.5993 max=1023 | onnx_max=1.02e+03 bmodel_max=1.04e+03 |
| yolo26n-pose | OK | cos=0.9995 max=227.5 | onnx_max=656 bmodel_max=644 |
| yolo26n-seg | OK | cos=0.9919 max=146.2 | onnx_max=638 bmodel_max=638 |
| yolo26n-sem | OK | cos=0.9943 max=4.237 | onnx_max=18 bmodel_max=18.2 |
| yolo26n-depth | OK | cos=0.9964 max=1.306 | onnx_max=5.46 bmodel_max=6.43 |
| ocr-det | OK | cos=0.9874 max=0.8707 | onnx_max=0.999 bmodel_max=1 |
| ocr-cls | OK | cos=0.9997 max=0.01709 | onnx_max=0.767 bmodel_max=0.784 |
| ocr-rec | OK | cos=0.9817 max=0.7628 | onnx_max=0.999 bmodel_max=0.999 |
| lpr-det | OK | cos=0.9999 max=102.6 | onnx_max=744 bmodel_max=642 |
| lpr-rec | OK | cos=0.9982 max=1.805 | onnx_max=31.1 bmodel_max=32.5 |
| scrfd | OK | cos=0.9921 max=0.3433 | onnx_max=0.236 bmodel_max=0.216 |
| age | OK | cos=0.9975 max=0.007609 | onnx_max=0.0914 bmodel_max=0.0902 |
| gender | OK | cos=0.9999 max=0.00817 | onnx_max=0.788 bmodel_max=0.796 |
| facerec | OK | cos=0.9385 max=5.091 | onnx_max=18.4 bmodel_max=18.8 |
| fas1 | OK | cos=1.0000 max=0.002149 | onnx_max=0.986 bmodel_max=0.988 |
| fas2 | 待转 | 取修复后 onnx 于 300×300 | Reshape 烘焙已修复, TRT/MNN 重转通过, 待 sophgo 容器补 bmodel |
| det10g | OK | cos=0.9922 max=0.4156 | onnx_max=0.388 bmodel_max=0.373 |
| 2d106 | OK | cos=0.9427 max=0.2363 | onnx_max=0.792 bmodel_max=0.862 |
| 1k3d | OK | cos=0.9992 max=0.01649 | onnx_max=0.306 bmodel_max=0.29 |
| w600k | OK | cos=0.8284 max=0.9125 | onnx_max=1.51 bmodel_max=1.29 |
| genderage | OK | cos=0.9987 max=0.04322 | onnx_max=0.366 bmodel_max=0.347 |
| zhgd-det | SKIP |  |  |
| zhgd-ml | OK | cos=0.9672 max=0.2978 | onnx_max=0.988 bmodel_max=0.984 |
