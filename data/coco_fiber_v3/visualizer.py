import os
import cv2
import random
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools.coco import COCO

IMAGE_DIR = r"D:\coding\fiberrcnn\data\raw"
coco = COCO("train.json")

MetadataCatalog.get("fiber_dataset").thing_classes = ["fiber"]

img_ids = coco.getImgIds()
for img_id in random.sample(img_ids, 3):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Impossible de charger {img_path}")
        continue

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Conversion COCO → Detectron2 format
    dataset_dict = {
        "file_name": img_path,
        "image_id": img_id,
        "height": img_info["height"],
        "width": img_info["width"],
        "annotations": []
    }
    for ann in anns:
        obj = {
            "bbox": ann["bbox"],
            "bbox_mode": BoxMode.XYWH_ABS,   # <--- obligatoire
            "category_id": ann["category_id"] -1,  # <--- obligatoire
            "segmentation": ann.get("segmentation", []),
            "keypoints": ann.get("keypoints", [])
        }
        dataset_dict["annotations"].append(obj)

    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("fiber_dataset"), scale=1.0)
    out = v.draw_dataset_dict(dataset_dict)

    plt.figure(figsize=(12,8))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis("off")
    plt.show()
