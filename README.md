# ST-DPEC

Implementation of **Open-Set SEM Defect Recognition via Shape–Texture Dual-Prototype Energy Calibration**. Separate mask-derived shape and masked texture spaces supply nearest-known prototype deviations, which are standardized and fused with classifier energy.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, torchvision 0.22.1, scikit-learn 1.6.1, matplotlib 3.10.0, pillow 12.3.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Provide SEM defect images, binary defect masks, class labels and original-instance identities. The manuscript's processed SEM/Fab collection is supplied by the dataset owner; this repository does not invent images or substitute table counts for a dataset. The same format accepts other masked defect collections.

Create `data/manifest.json` with paths relative to that file:

```json
[
  {"image": "images/a.png", "mask": "masks/a.png", "label": "Block", "group": "source_a", "split": "train"},
  {"image": "images/b.png", "mask": "masks/b.png", "label": "Block", "group": "source_b", "split": "dev"},
  {"image": "images/c.png", "mask": "masks/c.png", "label": "Block", "group": "source_c", "split": "test"},
  {"image": "images/d.png", "mask": "masks/d.png", "label": "Needle", "group": "source_d", "split": "test"}
]
```

Include every selected known class in training and development. Synthetic variants of the same source defect must share `group` and partition. Empty defect masks are rejected. Preparation resizes images and masks to 224×224 and removes every held-out class from fitting and calibration partitions.



For leave-one-class-out evaluation, repeat this sequence with each class held out and separate data/run directories. Use `--unknown ClassA ClassB` for two-class unknown protocols. Synthetic-to-Fab evaluation uses synthetic training/development rows and held-out real test rows, retaining the known/unknown taxonomy and source-instance separation. Domain names do not override class membership.

```bash
dualproto prepare --manifest data/manifest.json --unknown Needle --output data/prepared
```

## Training and inference

```bash
dualproto train --data data/prepared --output runs/train --seed 42
```

## Evaluation

```bash
dualproto evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

## Comparisons and ablations

```bash
dualproto evaluate --data data/prepared --checkpoint runs/train/model.pt --mode energy --output runs/energy
dualproto evaluate --data data/prepared --checkpoint runs/train/model.pt --mode shape --output runs/shape
dualproto evaluate --data data/prepared --checkpoint runs/train/model.pt --mode texture --output runs/texture
dualproto evaluate --data data/prepared --checkpoint runs/train/model.pt --mode shape_texture --output runs/shape_texture
dualproto evaluate --data data/prepared --checkpoint runs/train/model.pt --stress erode --output runs/erosion
dualproto evaluate --data data/prepared --checkpoint runs/train/model.pt --stress dilate --output runs/dilation
dualproto train --data data/prepared --seed 42 --prototype-weight 0 --output runs/no_proto
dualproto train --data data/prepared --seed 42 --energy-weight 0 --output runs/no_energy_margin
```

Evaluate each fitted control with its own checkpoint; scoring/stress controls above use the full checkpoint.

## Outputs

Training writes `runs/train/model.pt` with prototypes, standardization and fusion settings. Evaluation writes `metrics.json`, `scores.npz` and `scores.png`.
