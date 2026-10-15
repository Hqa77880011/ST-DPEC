# Shape–Texture Dual Prototypes

Implementation of **Open-Set SEM Defect Recognition via Shape–Texture Dual-Prototype Energy Calibration**. Separate mask-derived shape and masked texture spaces supply nearest-known prototype deviations, which are standardized and fused with classifier energy.

## Install and data

Use Python 3.10+ and activate a virtual environment:

```bash
python -m venv .venv
python -m pip install -e .
```

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

```bash
dualproto prepare --manifest data/manifest.json --unknown Needle --output data/prepared
dualproto train --data data/prepared --output runs/train
dualproto evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

For leave-one-class-out evaluation, repeat this sequence with each class held out and separate data/run directories. Use `--unknown ClassA ClassB` for two-class unknown protocols. Synthetic-to-Fab evaluation uses synthetic training/development rows and held-out real test rows, retaining the known/unknown taxonomy and source-instance separation. Domain names do not override class membership.

## Fitting and comparisons

The model uses ResNet-50, a 6→64→64 shape MLP, and a classifier over the joint shape/texture representation. `--pretrained` initializes ResNet-50 from TorchVision ImageNet weights. Other flags include `--epochs 100`, `--batch-size 32`, `--lr 0.0001`, `--prototype-weight 0.2`, `--energy-weight 0.1`, `--margin-in -8`, `--margin-out -4`, and `--seed`. Pass `--device cuda` for a compatible GPU installation.

Choose energy margins using development subjects/instances and freeze them before test evaluation. Prototype consistency uses detached class means recomputed each epoch; final prototypes are exact training-set means under the frozen final model. Energy temperature is one. Pseudo-unknowns use mask dilation, erosion, splitting and merging, combined with cross-class texture exchange where a different class is available in the batch. Real held-out classes never supply these examples.

Fusion weights are selected from the 0.1 simplex grid by known-versus-pseudo-unknown development AUROC. Score means and standard deviations come from known development inputs. The deployment threshold is the 95th percentile of known development scores and stays fixed for mask stress.

```bash
dualproto evaluate --mode energy --output runs/energy
dualproto evaluate --mode shape --output runs/shape
dualproto evaluate --mode texture --output runs/texture
dualproto evaluate --mode shape_texture --output runs/shape_texture
dualproto evaluate --stress erode --output runs/erosion
dualproto evaluate --stress dilate --output runs/dilation
dualproto train --prototype-weight 0 --output runs/no_proto
dualproto train --energy-weight 0 --output runs/no_energy_margin
```

The scoring controls share the trained checkpoint and prototypes. Evaluate training ablations using their own checkpoints. Mask stress uses a two-pixel erosion or dilation on the reference mask, with the clean-development threshold retained. Image pixels are unchanged.

## Outputs and metrics

Training writes `model.pt` with model weights, both prototype sets, standardization, fusion weights, class names and known-development components. Evaluation writes `metrics.json`, `scores.npz` and `scores.png`.

Metrics include AUROC, AUPR-OUT, OSCR, known-class macro-F1, known macro-F1 after rejection, validation-locked ID false rejection and OOD acceptance. Operating points are kept distinct:

- `fpr95_ood_detection` is ID false rejection at at least 95% OOD detection.
- `oar_at_95id` is OOD acceptance at the ID-test 95th-percentile curve threshold; its achieved ID acceptance is also reported.
- Validation-locked rates use the known-development threshold without consulting test OOD labels.

All ratios are fractions. OSCR integrates correct known acceptance against unknown acceptance, handling tied scores together. The test partition must contain both known and unknown inputs. Aggregate LOCO runs equally across held-out classes before aggregating seeds.

## Implementation choices

Shape descriptors summarize union-mask area, perimeter, major/minor-axis ratio, eccentricity, circularity and solidity. Area and perimeter are normalized by image area and square-root area to keep their units comparable across preprocessing. Texture uses nearest-resized mask pooling over the final backbone map; a zero mask produced by erosion yields a zero texture vector. Shape and texture deviations each minimize cosine distance over every known prototype, independently of the classifier's predicted label. Late fusion preserves those two separate compatibility tests until scoring. Pseudo-unknown morphology and texture operations are explicit training/calibration conventions, not samples from the real unknown distribution.
