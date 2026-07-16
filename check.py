import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import cv2
import matplotlib.pyplot as plt

# --- Charger ta config ---
cfg = get_cfg()
cfg.merge_from_file("configs/fiber_rcnn_r50_fpn.yaml")
cfg.MODEL.WEIGHTS = "output/run15_full_heads_orderfix/model_final.pth"  # mets ton checkpoint
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NAME = "FiberROIHeads"
predictor = DefaultPredictor(cfg)

# --- Charger une image annotée ---
img_path = "data/raw/1_0006.jpg"
image = cv2.imread(img_path)

# --- Forward pass ---
with torch.no_grad():
    outputs = predictor(image)

# --- Inspecter les masques bruts ---
if "pred_masks" in outputs["instances"].get_fields():
    masks = outputs["instances"].pred_masks.cpu().numpy()
    print("Masques shape:", masks.shape)
    print("Masque[0] stats:", masks[0].mean(), masks[0].min(), masks[0].max())

    plt.imshow(masks[0], cmap="gray")
    plt.title("Masque brut")
    plt.show()

# --- Inspecter les keypoints bruts ---
if "pred_keypoints" in outputs["instances"].get_fields():
    kps = outputs["instances"].pred_keypoints.cpu().numpy()
    print("Keypoints shape:", kps.shape)
    print("Exemple keypoints:", kps[0])

    # Visualiser sur l'image
    vis_img = image.copy()
    for (x, y, score) in kps[0]:
        cv2.circle(vis_img, (int(x), int(y)), 3, (0, 0, 255), -1)
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Keypoints bruts")
    plt.show()
