import numpy as np
import torch
from skimage.measure import regionprops
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


def shape_descriptor(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(6, np.float32)
    properties = regionprops(mask.astype(np.uint8))[0]
    area, perimeter = properties.area, properties.perimeter
    return np.array([area / mask.size, perimeter / np.sqrt(mask.size),
                     properties.axis_major_length / max(properties.axis_minor_length, 1),
                     properties.eccentricity, 4 * np.pi * area / max(perimeter ** 2, 1),
                     properties.solidity], np.float32)


class DualPrototypeNet(nn.Module):
    def __init__(self, classes, pretrained=False):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.shape = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64))
        self.classifier = nn.Linear(2112, classes)

    def forward(self, x, mask):
        descriptors = torch.as_tensor(np.stack([shape_descriptor(m) for m in mask.detach().cpu().numpy()[:, 0]]), device=x.device)
        shape = self.shape(descriptors)
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        feature = self.encoder((x - mean) / std)
        resized = F.interpolate(mask.float(), size=feature.shape[-2:], mode="nearest")
        texture = (feature * resized).sum((2, 3)) / resized.sum((2, 3)).clamp_min(1e-6)
        logits = self.classifier(torch.cat([shape, texture], -1))
        return logits, shape, texture


def pseudo_unknown(x, mask, labels, generator):
    changed_x, changed_mask = x.clone(), mask.clone()
    for i in range(len(x)):
        operation = int(torch.randint(4, (1,), generator=generator))
        if operation == 0:
            changed_mask[i:i + 1] = F.max_pool2d(mask[i:i + 1], 9, 1, 4)
        elif operation == 1:
            changed_mask[i:i + 1] = -F.max_pool2d(-mask[i:i + 1], 9, 1, 4)
        elif operation == 2:
            center = mask.shape[-1] // 2
            changed_mask[i, :, :, center - 3:center + 3] = 0
        else:
            shifted = F.pad(mask[i:i + 1, :, :, :-12], (12, 0, 0, 0))
            changed_mask[i:i + 1] = torch.maximum(mask[i:i + 1], shifted)
        alternatives = torch.nonzero(labels != labels[i], as_tuple=True)[0]
        if len(alternatives):
            j = int(alternatives[i % len(alternatives)])
            changed_x[i] = torch.where(mask[i].bool(), x[j], x[i])
    return changed_x, changed_mask


def score_components(logits, shape, texture, shape_prototypes, texture_prototypes, temperature=1):
    energy = -temperature * torch.logsumexp(logits / temperature, -1)
    shape_distance = 1 - F.normalize(shape, dim=-1) @ F.normalize(shape_prototypes, dim=-1).T
    texture_distance = 1 - F.normalize(texture, dim=-1) @ F.normalize(texture_prototypes, dim=-1).T
    return torch.stack([energy, shape_distance.min(-1).values, texture_distance.min(-1).values], -1)
