import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn import functional as F

from .method import DualPrototypeNet, pseudo_unknown, score_components
from .metrics import detection_metrics


def prepare(args):
    manifest = Path(args.manifest)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    groups, samples = {}, {}
    known = sorted({r["label"] for r in rows if r["label"] not in args.unknown})
    if not known or not args.unknown:
        raise ValueError("Declare known and held-out unknown labels")
    for row in rows:
        group, split = row["group"], row["split"]
        if group in groups and groups[group] != split:
            raise ValueError("Variants of one source instance cross partitions")
        groups[group] = split
        is_unknown = row["label"] in args.unknown
        if is_unknown and split != "test":
            continue
        root = manifest.resolve().parent
        image = np.asarray(Image.open(root / row["image"]).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), np.float32) / 255
        mask = np.asarray(Image.open(root / row["mask"]).convert("L").resize((224, 224), Image.Resampling.NEAREST)) > 0
        if not mask.any():
            raise ValueError("Defect recognition requires a nonempty defect mask")
        label = -1 if is_unknown else known.index(row["label"])
        samples.setdefault(split, []).append((image.transpose(2, 0, 1), mask, label, str(group)))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for split, values in samples.items():
        x, mask, y, group = zip(*values)
        np.savez_compressed(out / f"{split}.npz", x=np.array(x), mask=np.array(mask), y=y, group=group)
    (out / "classes.json").write_text(json.dumps({"known": known, "unknown": args.unknown}, indent=2))


def tensors(data, index, device, stress="none"):
    x = torch.as_tensor(data["x"][index], device=device)
    mask = torch.as_tensor(data["mask"][index], dtype=torch.float32, device=device)[:, None]
    if stress == "erode":
        mask = -F.max_pool2d(-mask, 5, 1, 2)
    elif stress == "dilate":
        mask = F.max_pool2d(mask, 5, 1, 2)
    y = torch.as_tensor(data["y"][index], dtype=torch.long, device=device)
    return x, mask, y


def extract(net, data, args, pseudo=False):
    outputs, shapes, textures, labels = [], [], [], []
    generator = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for start in range(0, len(data["y"]), args.batch_size):
            index = np.arange(start, min(start + args.batch_size, len(data["y"])))
            x, mask, y = tensors(data, index, args.device, getattr(args, "stress", "none"))
            if pseudo:
                x, mask = pseudo_unknown(x, mask, y, generator)
            logits, shape, texture = net(x, mask)
            outputs.append(logits)
            shapes.append(shape)
            textures.append(texture)
            labels.append(y)
    return tuple(torch.cat(values) for values in [outputs, shapes, textures, labels])


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_data, development = [np.load(Path(args.data) / f"{split}.npz", allow_pickle=False) for split in ["train", "dev"]]
    classes = json.loads((Path(args.data) / "classes.json").read_text())["known"]
    if np.any(train_data["y"] < 0) or np.any(development["y"] < 0):
        raise ValueError("Real unknowns cannot enter fitting or calibration")
    net = DualPrototypeNet(len(classes), args.pretrained).to(args.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.epochs):
        net.eval()
        _, shape, texture, labels = extract(net, train_data, args)
        ps = torch.stack([shape[labels == k].mean(0) for k in range(len(classes))]).detach()
        pt = torch.stack([texture[labels == k].mean(0) for k in range(len(classes))]).detach()
        if not torch.isfinite(ps).all():
            raise ValueError("Every known class requires training samples")
        net.train()
        order = rng.permutation(len(train_data["y"]))
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            x, mask, y = tensors(train_data, index, args.device)
            px, pm = pseudo_unknown(x, mask, y, generator)
            logits, shape, texture = net(x, mask)
            pseudo_logits = net(px, pm)[0]
            proto_loss = (1 - F.cosine_similarity(shape, ps[y])).mean() + (1 - F.cosine_similarity(texture, pt[y])).mean()
            energy = -torch.logsumexp(logits, -1)
            pseudo_energy = -torch.logsumexp(pseudo_logits, -1)
            loss = F.cross_entropy(logits, y) + args.prototype_weight * proto_loss
            loss = loss + args.energy_weight * (F.relu(energy - args.margin_in).mean() + F.relu(args.margin_out - pseudo_energy).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    net.eval()
    _, shape, texture, labels = extract(net, train_data, args)
    ps = torch.stack([shape[labels == k].mean(0) for k in range(len(classes))])
    pt = torch.stack([texture[labels == k].mean(0) for k in range(len(classes))])
    log, shape, texture, _ = extract(net, development, args)
    plog, pshape, ptexture, _ = extract(net, development, args, True)
    id_components = score_components(log, shape, texture, ps, pt).cpu().numpy()
    pseudo_components = score_components(plog, pshape, ptexture, ps, pt).cpu().numpy()
    mean, scale = id_components.mean(0), np.maximum(id_components.std(0), 1e-6)
    validation = (np.concatenate([id_components, pseudo_components]) - mean) / scale
    unknown = np.r_[np.zeros(len(id_components)), np.ones(len(pseudo_components))]
    grid = [np.array([e, s, 10 - e - s]) / 10 for e in range(11) for s in range(11 - e)]
    weights = grid[int(np.argmax([roc_auc_score(unknown, validation @ w) for w in grid]))]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "shape_prototypes": ps.cpu(), "texture_prototypes": pt.cpu(),
                "mean": torch.tensor(mean), "scale": torch.tensor(scale), "weights": torch.tensor(weights),
                "validation_components": torch.tensor(id_components), "classes": classes}, out / "model.pt")


def evaluate(args):
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    net = DualPrototypeNet(len(saved["classes"])).to(args.device)
    net.load_state_dict(saved["model"])
    net.eval()
    data = np.load(Path(args.data) / "test.npz", allow_pickle=False)
    logits, shape, texture, labels = extract(net, data, args)
    components = score_components(logits, shape, texture, saved["shape_prototypes"], saved["texture_prototypes"])
    validation = saved["validation_components"]
    weights = saved["weights"]
    if args.mode != "full":
        weights = logits.new_tensor({"energy": [1, 0, 0], "shape": [0, 1, 0], "texture": [0, 0, 1], "shape_texture": [0, 0.5, 0.5]}[args.mode])
    score = (((components - saved["mean"]) / saved["scale"]) @ weights).cpu().numpy()
    vscore = (((validation - saved["mean"]) / saved["scale"]) @ weights).cpu().numpy()
    threshold = float(np.quantile(vscore, 0.95))
    predicted, labels = logits.argmax(1).cpu().numpy(), labels.cpu().numpy()
    known = labels >= 0
    if known.all() or not known.any():
        raise ValueError("Test partition needs both known and held-out unknown defects")
    metrics = detection_metrics(score[known], score[~known], threshold, labels[known], predicted[known])
    metrics["known_macro_f1"] = float(f1_score(labels[known], predicted[known], labels=range(len(saved["classes"])), average="macro", zero_division=0))
    rejected = predicted.copy()
    rejected[score > threshold] = -1
    metrics["known_with_rejection_macro_f1"] = float(f1_score(labels[known], rejected[known], labels=range(len(saved["classes"])), average="macro", zero_division=0))
    metrics["threshold"] = threshold
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    np.savez_compressed(out / "scores.npz", score=score, predicted=predicted, y=labels, components=components.cpu().numpy())
    fig, ax = plt.subplots(figsize=(6, 4))
    for mask, name in [(known, "Known"), (~known, "Unknown")]:
        ax.hist(score[mask], bins=30, density=True, histtype="step", label=name)
    ax.axvline(threshold, color="gray", linestyle="--")
    ax.legend()
    ax.set(xlabel="Unknown score", ylabel="Density")
    fig.tight_layout()
    fig.savefig(out / "scores.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", required=True)
    p.add_argument("--unknown", nargs="+", required=True)
    p.add_argument("--output", default="data/prepared")
    p.set_defaults(function=prepare)
    for command in ["train", "evaluate"]:
        p = sub.add_parser(command)
        p.add_argument("--data", default="data/prepared")
        p.add_argument("--output", default=f"runs/{command}")
        p.add_argument("--device", default="cpu")
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--seed", type=int, default=42)
        if command == "train":
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--prototype-weight", type=float, default=0.2)
            p.add_argument("--energy-weight", type=float, default=0.1)
            p.add_argument("--margin-in", type=float, default=-8)
            p.add_argument("--margin-out", type=float, default=-4)
            p.add_argument("--pretrained", action="store_true")
            p.set_defaults(function=train)
        else:
            p.add_argument("--checkpoint", default="runs/train/model.pt")
            p.add_argument("--mode", choices=["full", "energy", "shape", "texture", "shape_texture"], default="full")
            p.add_argument("--stress", choices=["none", "erode", "dilate"], default="none")
            p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)
