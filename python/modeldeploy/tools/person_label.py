"""按 zhgd_det 模型把图片按"是否有人"分目录，并生成 labelme(json) 标注。

用法（安装 modeldeploy wheel 后）:
    md-person-label --src <图片目录> --people <有人的目录> --nopeople <没人的目录> \\
                    --model <zhgd_det_20251219.onnx> [--conf 0.25] [--batch 16]

或作为库:
    from modeldeploy.tools.person_label import label_dataset
    stats = label_dataset(src, people_dir, nopeople_dir, model_path)
"""

import argparse
import json
import os
import shutil
import sys

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

LABELME_TEMPLATE = {
    "version": "3.2.1",
    "flags": {},
    "imagePath": None,
    "imageData": None,
    "imageHeight": 0,
    "imageWidth": 0,
    "shapes": [],
}


def box_to_points(x, y, w, h):
    """Rect2f (x,y,w,h) -> labelme 矩形 4 角点序列。"""
    return [[float(x), float(y)],
            [float(x + w), float(y)],
            [float(x + w), float(y + h)],
            [float(x), float(y + h)]]


def build_labelme(basename, height, width, detections):
    """按 F:\\zhgd\\Detection\\dataset 的 labelme 格式生成标注 dict。"""
    doc = json.loads(json.dumps(LABELME_TEMPLATE))
    doc["imagePath"] = basename
    doc["imageHeight"] = int(height)
    doc["imageWidth"] = int(width)
    shapes = []
    for det in detections:
        box = det.box  # Rect2f: x, y, width, height
        shapes.append({
            "label": "person",
            "score": det.score,
            "points": box_to_points(box.x, box.y, box.width, box.height),
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {},
            "kie_linking": [],
        })
    doc["shapes"] = shapes
    return doc


def label_dataset(src_dir, people_dir, nopeople_dir, model_path,
                  conf=0.25, size=1280, keep_labels=None):
    """对 src_dir 内图片推理，写 labelme json 并把图片移动到对应分类目录。

    keep_labels: 只统计/标注这些 label_id（COCO person=0），其余类别忽略。
    返回 {'people': n, 'nopeople': n, 'skipped': [files]}
    """
    import cv2
    import numpy as np
    import modeldeploy

    os.makedirs(people_dir, exist_ok=True)
    os.makedirs(nopeople_dir, exist_ok=True)

    option = modeldeploy.RuntimeOption()
    model = modeldeploy.vision.UltralyticsDet(str(model_path), option)
    model.postprocessor.conf_threshold = conf
    model.preprocessor.size = [size, size]  # 静态 batch=1

    files = sorted(
        f for f in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, f))
        and os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    total = len(files)
    stats = {"people": 0, "nopeople": 0, "skipped": []}

    for i, name in enumerate(files, 1):
        path = os.path.join(src_dir, name)
        # Windows 下 cv2.imread 无法打开含中文路径，改用 imdecode + np.fromfile
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            stats["skipped"].append(name)
            continue
        h, w = img.shape[:2]
        dets = model.predict(img)  # 静态 batch=1，逐张推理
        if keep_labels:
            dets = [d for d in dets if d.label_id in keep_labels]
        has_person = any(d.score >= conf for d in dets)
        target_dir = people_dir if has_person else nopeople_dir
        doc = build_labelme(name, h, w, dets)
        with open(os.path.join(target_dir, name + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        shutil.move(path, os.path.join(target_dir, name))
        if has_person:
            stats["people"] += 1
        else:
            stats["nopeople"] += 1
        print(f"[{i}/{total}] "
              f"有人的={stats['people']} 没人的={stats['nopeople']}", flush=True)
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="按检测模型把图片按是否有人分目录并生成 labelme json 标注。")
    parser.add_argument("--src", required=True, help="待处理的图片目录")
    parser.add_argument("--people", required=True, help="'有人的' 输出目录")
    parser.add_argument("--nopeople", required=True, help="'没人的' 输出目录")
    parser.add_argument("--model", required=True, help="检测 onnx 模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--size", type=int, default=1280, help="letterbox 输入尺寸")
    parser.add_argument("--classes", default="0",
                        help="只统计/标注这些类别的逗号分隔 label_id（默认 0=person）")
    args = parser.parse_args(argv)

    keep = {int(x) for x in args.classes.split(",") if x.strip()}
    stats = label_dataset(args.src, args.people, args.nopeople,
                          args.model, args.conf, args.size, keep)
    print("\n完成: 有人的=%d 没人的=%d 跳过=%d" %
          (stats["people"], stats["nopeople"], len(stats["skipped"])))
    if stats["skipped"]:
        print("跳过的文件:", stats["skipped"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
