# FiberRCNN v2

**Advanced Scientific Nanofiber Analysis Framework**

FiberRCNN v2 is a production-grade, open-source deep learning framework for automated characterization of electrospun nanofibers in SEM images. It extends Detectron2's Mask R-CNN with nine specialized prediction heads and a deterministic morphological analysis pipeline.

---

## Features

| Module | Capability |
|---|---|
| **Data** | LabelMe → COCO-Fiber converter with auto-generated centerlines, keypoints, and geometric features |
| **Geometry** | Polygon → mask → skeleton → graph → centerline → 40 keypoints → width/length/curvature/orientation/tortuosity |
| **Model** | FiberRCNN with 9 heads: box, mask (BCE+Dice), keypoints, width, length, curvature, orientation (circular loss), tortuosity, quality |
| **Backbones** | ResNet-50/101-FPN (default), ConvNeXt-Tiny, Swin-T, Swin-S |
| **Training** | AMP, DDP, gradient accumulation, early stopping, TensorBoard, W&B |
| **Morphology** | Post-processing: porosity, coverage, fiber density, alignment score, pore size distribution, intersection count |
| **Evaluation** | AP/AP50/AP75, mask mAP, OKS, PCK, MAE/RMSE for all regression targets, angular error |
| **Visualization** | Instance overlays, centerline maps, width/orientation/pore maps, rose plots, histograms |
| **Export** | ONNX export (backbone+FPN) for TensorRT deployment |

---

## Installation

### 1. Prerequisites

```bash
# Python 3.11+, CUDA 12+
conda create -n fiberrcnn python=3.11
conda activate fiberrcnn
```

### 2. Install PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Detectron2

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 4. Install FiberRCNN

```bash
git clone https://github.com/your-org/fiberrcnn.git
cd fiberrcnn
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Prepare annotations

Label your SEM images with [LabelMe](https://github.com/wkentaro/labelme) using `polygon` shapes with label `fiber`.

```
/data/raw/
    sem_001.json    ← LabelMe annotation
    sem_001.png
    sem_002.json
    sem_002.png
    ...
```

### 2. Convert to COCO-Fiber

```bash
python tools/convert_dataset.py \
    --labelme_dir /data/raw \
    --output_dir  /data/coco_fiber \
    --split \
    --train_ratio 0.8 \
    --val_ratio   0.1
```

This generates `/data/coco_fiber/{train,val,test}.json` with segmentation masks, 40 ordered keypoints, and all geometric attributes per fiber.

### 3. Verify the conversion

```bash
python tools/convert_dataset.py --verify /data/coco_fiber/train.json
```

### 4. Train

```bash
python tools/train.py \
    --config     configs/fiber_rcnn_r50_fpn.yaml \
    --train_json /data/coco_fiber/train.json \
    --val_json   /data/coco_fiber/val.json \
    --image_root /data/images \
    --output_dir ./output/run01 \
    --wandb_project my_nanofiber_project
```

Multi-GPU (4 GPUs):
```bash
python -m torch.distributed.run --nproc_per_node=4 tools/train.py \
    --config configs/fiber_rcnn_r50_fpn.yaml \
    --train_json /data/coco_fiber/train.json \
    --val_json   /data/coco_fiber/val.json \
    --image_root /data/images \
    --output_dir ./output/run01
```

Resume from checkpoint:
```bash
python tools/train.py ... --resume
```

### 5. Inference

```bash
# Single image
python tools/infer.py \
    --config  configs/fiber_rcnn_r50_fpn.yaml \
    --weights output/run01/model_final.pth \
    --input   /data/test/sem_042.png \
    --output_dir ./results

# Directory
python tools/infer.py \
    --config     configs/fiber_rcnn_r50_fpn.yaml \
    --weights    output/run01/model_final.pth \
    --input_dir  /data/test_images \
    --output_dir ./results \
    --threshold  0.6
```

### 6. Evaluate

```bash
python tools/evaluate.py \
    --config     configs/fiber_rcnn_r50_fpn.yaml \
    --weights    output/run01/model_final.pth \
    --test_json  /data/coco_fiber/test.json \
    --image_root /data/images \
    --output_dir ./eval_results
```

### 7. Export to ONNX

```bash
python tools/export_onnx.py \
    --config  configs/fiber_rcnn_r50_fpn.yaml \
    --weights output/run01/model_final.pth \
    --output  output/fiberrcnn_backbone.onnx \
    --dynamic
```

---

## Python API

```python
import fiberrcnn
from fiberrcnn.engine.inference import FiberPredictor

predictor = FiberPredictor.from_config(
    "configs/fiber_rcnn_r50_fpn.yaml",
    "output/run01/model_final.pth",
    score_thresh=0.5,
)

result = predictor.predict("sem_image.png")

# Per-fiber results
for fiber in result.fiber_instances:
    print(f"Fiber {fiber.instance_id}: "
          f"width={fiber.fiber_width:.1f}px  "
          f"length={fiber.fiber_length:.1f}px  "
          f"orientation={fiber.fiber_orientation:.1f}°  "
          f"tortuosity={fiber.fiber_tortuosity:.3f}  "
          f"confidence={fiber.confidence:.2f}")

# Image-level metrics
m = result.image_metrics
print(f"Porosity:        {m['porosity']:.3f}")
print(f"Coverage:        {m['coverage_ratio']:.3f}")
print(f"Fiber density:   {m['fiber_density']:.2f} fibers/10000px²")
print(f"Alignment score: {m['alignment_score']:.3f}")
print(f"Mean pore size:  {m['mean_pore_size']:.1f}px")

# Save JSON
result.save_json("results/sem_image_prediction.json")
```

### Batch inference

```python
from pathlib import Path

images = list(Path("/data/test").glob("*.png"))
results = predictor.predict_batch(images, output_dir="results/", save_json=True)
```

### Geometry pipeline

```python
from fiberrcnn.geometry import compute_fiber_geometry

geom = compute_fiber_geometry(
    points=[[10,47],[110,47],[110,52],[10,52]],
    image_height=100,
    image_width=120,
)

print(geom.fiber_width)       # ~5.0 px
print(geom.fiber_length)      # ~100.0 px
print(geom.fiber_orientation) # ~0.0° (horizontal)
print(geom.fiber_tortuosity)  # ~1.0 (straight)
print(geom.keypoints.shape)   # (40, 3)
```

### Morphological analysis

```python
from fiberrcnn.morphology import compute_image_morphology

morph = compute_image_morphology(
    masks=masks,
    centerlines=centerlines,
    widths=widths,
    lengths=lengths,
    curvatures=curvatures,
    orientations=orientations,
    tortuosities=tortuosities,
    image_height=H,
    image_width=W,
)

print(morph.porosity)
print(morph.alignment_score)
print(morph.pore_stats.mean_pore_size)
```

---

## Output Format

### Per-fiber JSON

```json
{
  "instance_id": 0,
  "bbox": [10.0, 47.0, 100.0, 5.0],
  "confidence": 0.97,
  "fiber_width": 5.2,
  "fiber_length": 98.4,
  "fiber_curvature": 0.0012,
  "fiber_orientation": 2.3,
  "fiber_tortuosity": 1.008,
  "has_bead": false,
  "is_blurry": false,
  "is_crossing": false,
  "keypoints": [[10.2, 49.8], [12.7, 50.1], "..."]
}
```

### Image-level JSON

```json
{
  "porosity": 0.73,
  "coverage_ratio": 0.27,
  "fiber_density": 4.2,
  "alignment_score": 0.88,
  "intersection_count": 3,
  "junction_density": 0.3,
  "mean_fiber_width": 5.8,
  "mean_fiber_length": 112.4,
  "mean_curvature": 0.0014,
  "mean_tortuosity": 1.012,
  "mean_pore_size": 24.6,
  "median_pore_size": 21.3,
  "max_pore_size": 61.0,
  "pore_size_std": 8.4,
  "pore_count": 47
}
```

---

## Backbone Selection

Change the backbone in the config file:

```yaml
# ResNet-50-FPN (default, fastest)
MODEL:
  BACKBONE:
    NAME: "build_resnet_fpn_backbone"
  RESNETS:
    DEPTH: 50

# ResNet-101-FPN (more accurate)
  RESNETS:
    DEPTH: 101

# ConvNeXt-Tiny (modern, competitive)
MODEL:
  BACKBONE:
    NAME: "build_convnext_tiny_fpn_backbone"

# Swin-T (transformer, highest accuracy)
MODEL:
  BACKBONE:
    NAME: "build_swin_t_fpn_backbone"

# Swin-S (larger transformer)
MODEL:
  BACKBONE:
    NAME: "build_swin_s_fpn_backbone"
```

---

## COCO-Fiber Format

The COCO-Fiber format extends the standard COCO annotation schema:

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [x, y, w, h],
  "area": 523,
  "segmentation": [[x0,y0,x1,y1,...]],
  "iscrowd": 0,
  "num_keypoints": 40,
  "keypoints": [x0,y0,v0, x1,y1,v1, ...],
  "fiber_width": 5.23,
  "fiber_length": 98.7,
  "fiber_curvature": 0.00124,
  "fiber_orientation": 12.4,
  "fiber_tortuosity": 1.018
}
```

Keypoints represent the ordered centerline (from one fiber tip to the other), with visibility=2 for all labeled points.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_geometry.py -v
pytest tests/test_morphology.py -v
pytest tests/test_losses.py -v
pytest tests/test_heads.py -v
pytest tests/test_converter.py -v

# With coverage
pytest tests/ --cov=fiberrcnn --cov-report=html
```

---

## Project Structure

```
fiberrcnn/
├── configs/
│   ├── base_fiber_rcnn.yaml
│   ├── fiber_rcnn_r50_fpn.yaml
│   ├── fiber_rcnn_r101_fpn.yaml
│   ├── fiber_rcnn_convnext_tiny.yaml
│   ├── fiber_rcnn_swin_t.yaml
│   └── fiber_rcnn_swin_s.yaml
│
├── fiberrcnn/
│   ├── data/
│   │   ├── converter.py          # LabelMe → COCO-Fiber
│   │   └── dataset_mapper.py     # Detectron2 dataset registration + mapper
│   │
│   ├── geometry/
│   │   └── fiber_geometry.py     # Full geometry pipeline (13 steps)
│   │
│   ├── morphology/
│   │   └── fiber_morphology.py   # Post-processing image-level metrics
│   │
│   ├── modeling/
│   │   ├── heads/
│   │   │   └── fiber_heads.py    # 8 prediction heads
│   │   ├── roi_heads/
│   │   │   └── fiber_roi_heads.py  # FiberROIHeads (Detectron2 registered)
│   │   ├── losses/
│   │   │   └── fiber_losses.py   # Dice, Circular, Keypoint, Quality losses
│   │   ├── backbones/
│   │   │   └── fiber_backbones.py  # ConvNeXt + Swin registration
│   │   └── meta_arch/
│   │
│   ├── engine/
│   │   ├── trainer.py            # FiberTrainer + hooks (AMP, DDP, W&B, ES)
│   │   └── inference.py          # FiberPredictor + structured output
│   │
│   ├── evaluation/
│   │   └── fiber_evaluator.py    # All metrics
│   │
│   ├── visualization/
│   │   └── fiber_viz.py          # All visualisations
│   │
│   └── utils/
│       ├── misc.py               # Reproducibility, checkpointing, helpers
│       └── logging.py            # Loguru + TensorBoard wrappers
│
├── tools/
│   ├── convert_dataset.py
│   ├── train.py
│   ├── infer.py
│   ├── evaluate.py
│   └── export_onnx.py
│
├── tests/
│   ├── conftest.py
│   ├── test_geometry.py
│   ├── test_morphology.py
│   ├── test_losses.py
│   ├── test_heads.py
│   ├── test_converter.py
│   └── test_visualization.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## SEM Scale Calibration

Convert pixel measurements to physical units:

```python
from fiberrcnn.utils.misc import pixels_to_nm

nm_per_pixel = 10.5  # from SEM scale bar

for fiber in result.fiber_instances:
    width_nm  = pixels_to_nm(fiber.fiber_width,  nm_per_pixel)
    length_nm = pixels_to_nm(fiber.fiber_length, nm_per_pixel)
    print(f"Width: {width_nm:.1f} nm  Length: {length_nm:.1f} nm")
```

---

## Citation

If you use FiberRCNN v2 in your research, please cite:

```bibtex
@software{fiberrcnn2024,
  title   = {FiberRCNN v2: Advanced Scientific Nanofiber Analysis Framework},
  year    = {2024},
  url     = {https://github.com/your-org/fiberrcnn},
  version = {2.0.0}
}
```

---

## License

Apache License 2.0
