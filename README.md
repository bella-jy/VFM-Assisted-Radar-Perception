# VFM-Assisted-Radar-Perception
# SAM-Guided CenterFusion

A research prototype for 3D object detection under low visibility using camera images, radar data, and semantic priors from an adapted Segment Anything Model (SAM).

## Overview

- **Degradation-adaptive SAM:** generates semantic masks from low-visibility images and aligned pseudo-clear reference images.
- **Semantic-guided radar filtering:** estimates point reliability using semantic support, center proximity, and local radar consistency.
- **Semantic-gated Fusion:** combines image, semantic, and radar features to predict 3D objects.

SAM and the detector are trained separately. Semantic priors are generated offline before detector training or inference.

```text
Images + reference images -> Adapted SAM -> Semantic priors
Radar + calibration + semantic priors -> Filtered radar
Images + semantic priors + filtered radar -> CenterFusion -> 3D detections
```
## Installation

Use Python 3.10-3.12 and install matching `torch` and `torchvision` builds for your hardware first.

```bash
pip install -r requirements-semantic.txt
python -m pytest tests/test_semantic_pipeline.py -q
```

## Prepare Data

Prepare synchronized images and radar points, camera intrinsics, and radar-to-camera extrinsics. Transform radar points and velocities into camera coordinates before creating the detection JSON files.

```text
data/lowvis/
  images/             # Original images
  references/         # Aligned reference images with matching filenames
  instance_masks/     # Binary object masks for SAM training
  sam_train.json      # SAM training manifest
  train.json          # Detection training data
  val.json            # Detection validation data
  test.json           # Detection inference data
  priors/             # Generated semantic priors

models/
  sam_vit_b_01ec64.pth # SAM ViT-B checkpoint, supplied separately
```

SAM training requires binary instance masks. Detector training requires extended COCO annotations with classes, boxes, depth, dimensions, orientation, and velocity labels where available. Radar input uses the existing CenterFusion `18 x N` format, not raw CSV/PCD files. Split data by recording sequence.

## Run the Pipeline

### 1. Prepare reference images

Supply aligned pseudo-clear images in `data/lowvis/references/`. They are required for both SAM training and prior generation. If you have an external TorchScript restoration model, you can run:

```bash
python src/tools/low_visibility_sam.py references --images data/lowvis/images --restorer models/restorer.ts --output data/lowvis/references --device cuda
```

The diffusion restoration model itself is not implemented or provided here. Skip this command if reference images are already available.

### 2. Fine-tune SAM

```bash
python src/tools/low_visibility_sam.py train --manifest data/lowvis/sam_train.json --checkpoint models/sam_vit_b_01ec64.pth --output models/lowvis_sam --epochs 20 --device cuda
```

Output: `models/lowvis_sam/adapter_last.pth`.

### 3. Generate semantic priors

```bash
python src/tools/low_visibility_sam.py generate --images data/lowvis/images --references data/lowvis/references --checkpoint models/sam_vit_b_01ec64.pth --adapter models/lowvis_sam/adapter_last.pth --output data/lowvis/priors --device cuda
```

Each image produces an original-resolution `.npz` prior and a mask preview. For example, `sequence01/frame001.jpg` produces `sequence01/frame001.jpg.npz`.

### 4. Train the detector

This example trains a one-class detector from scratch. Adjust the class count and batch size for your dataset and GPU memory.

```bash
python src/main.py ddd --dataset custom --num_classes 1 --custom_dataset_img_path data/lowvis/images --custom_dataset_ann_path "data/lowvis/{split}.json" --input_h 1024 --input_w 1024 --pointcloud --semantic_fusion --semantic_prior_dir data/lowvis/priors --velocity --dla_node conv --no_pretrained_backbone --exp_id semantic_lowvis --batch_size 6 --lr 0.0002 --num_epochs 60 --val_intervals 1 --num_workers 0 --shuffle_train --not_rand_crop --flip 0.5 --gpus 0
```

Output: `exp/ddd/semantic_lowvis/model_last.pth`.

### 5. Run 3D detection

```bash
python src/predict_semantic.py ddd --dataset custom --num_classes 1 --custom_dataset_img_path data/lowvis/images --custom_dataset_ann_path "data/lowvis/{split}.json" --val_split test --input_h 1024 --input_w 1024 --pointcloud --semantic_fusion --semantic_prior_dir data/lowvis/priors --velocity --dla_node conv --load_model exp/ddd/semantic_lowvis/model_last.pth --exp_id semantic_predict --gpus 0
```

Output: `exp/ddd/semantic_predict/semantic_detections_3d.json`, containing classes, scores, 3D positions, dimensions, orientations, velocities, and box corners. Add `--debug 4` to save visualizations.

For new data with trained models, repeat steps **1, 3, and 5**. Ground-truth object annotations are not required for inference. For CPU debugging, use `--device cpu` in SAM commands and `--gpus -1` in detector commands.


