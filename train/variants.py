"""Task 4 — the Task 3 pipeline with ONE switch, so each row is one change.

The baseline is the model that is on the phone now: the pretrained
`mobilenet_v3_small` of Task 2 with its last layer replaced, the whole network
trained with Adam(lr=1e-3) and CrossEntropyLoss for 15 epochs on the group
photographs — the cells 40 to 44 of the notebook, unchanged. Every other
variant is that baseline with exactly one thing added or swapped, named in the
first column of results.csv, so the table reads as "what did this one change
bring". A combination is a list of names: `--variant aug,cosine`.

    python3 train/variants.py --list
    python3 train/variants.py --variant baseline --seed 0
    python3 train/variants.py --variant aug --seeds 0,1,2
    python3 train/variants.py --variant all --seeds 0,1,2
    python3 train/variants.py --summary

Run it from the repository root. Everything below is the pipeline of Task 3:
LeafDataset from cell 54, the network of cell 56, the training loop of cell 44.
No framework, no helpers nobody asked for.
"""
import argparse
import copy
import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

device = "mps" if torch.backends.mps.is_available() else "cpu"
mean = [0.485, 0.456, 0.406]           # the ImageNet numbers of cell 40
std = [0.229, 0.224, 0.225]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data")
GROUP = os.path.join(DATA, "leaves")                 # 107 train / 29 val, your photos
HISTORY = os.path.join(DATA, "leaves_history")       # 700 train / 120 val, last year
CAMVAL = os.path.join(DATA, "camval")                # 232, made by make_camera_val.py
RESULTS = os.path.join(HERE, "results.csv")
WEIGHTS = os.path.join(HERE, "weights")          # every run saves its weights here
CLASSES = ["apple", "cherry", "chestnut", "maple"]


# ---------------------------------------------------------------- the data


class LeafDataset(Dataset):        # cell 54, as you wrote it
    def __init__(self, folder, transform):
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(folder, cls_name)
            files = sorted([os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                            if os.path.isfile(os.path.join(cls_dir, f))])
            for path in files:
                self.samples.append((path, class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, target = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def base_transform():              # cell 40: to tensor, standardize
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def train_transform(s):
    """The same, with augmentation in front when the variant asks for it."""
    steps = []
    if s["geom"]:                  # the geometric part of cell 60
        steps += [transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
                  transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                  transforms.RandomRotation(20)]
    if s["colour"]:                # the new part: colour, the thing a camera changes
        steps += [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                         saturation=0.4, hue=0.1)]
    return transforms.Compose(steps + base_transform().transforms)


def train_dataset(s):
    if s["data"] == "group":
        return LeafDataset(os.path.join(GROUP, "train"), train_transform(s))
    if s["data"] == "history":
        return LeafDataset(os.path.join(HISTORY, "train"), train_transform(s))
    return ConcatDataset([LeafDataset(os.path.join(GROUP, "train"), train_transform(s)),
                          LeafDataset(os.path.join(HISTORY, "train"), train_transform(s))])


# ---------------------------------------------------------------- the nets


def make_mobilenet(s):             # cell 42
    model = models.mobilenet_v3_small(weights="DEFAULT")
    if s["dropout"] is not None:
        model.classifier[2] = nn.Dropout(s["dropout"])     # shipped value is 0.2
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    if s["freeze"]:                # train the new last layer only
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.classifier[3].parameters():
            p.requires_grad_(True)
    return model.to(device)


class LeafNet(nn.Module):          # cell 56, two switches added at the end
    def __init__(self, gap=False, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1) if gap else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc1 = nn.Linear(64 if gap else 64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, len(CLASSES))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        if self.gap is not None:
            x = self.gap(x).flatten(1)      # 64 numbers, not 64*6*6
        else:
            x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc2(x)


def make_leafnet(s):
    return LeafNet(gap=s["gap"], dropout=s["dropout"] or 0.0).to(device)


# ------------------------------------------------------- the one switch

# Every variant below is the baseline with these keys changed and nothing else.
BASE = dict(net="mobilenet", data="group", epochs=15, geom=False, colour=False,
            freeze=False, dropout=None, gap=False, cosine=False,
            optimizer="adam", weight_decay=0.0, smoothing=0.0, best_epoch=False)

VARIANTS = {
    "baseline":      {},                                                  # Task 2 as it is
    "colour":        {"colour": True},                                    # ColorJitter
    "geom":          {"geom": True},                                      # cell 60 cropping
    "aug":           {"geom": True, "colour": True},                      # both
    "frozen":        {"freeze": True},                                    # head only
    "cosine":        {"cosine": True},                                    # CosineAnnealingLR
    "epochs30":      {"epochs": 30},                                      # train longer
    "adamw":         {"optimizer": "adamw", "weight_decay": 1e-4},        # AdamW + decay
    "smooth":        {"smoothing": 0.1},                                  # label smoothing
    "dropout50":     {"dropout": 0.5},                                    # 0.2 -> 0.5
    "bestepoch":     {"best_epoch": True},                                # keep best weights
    "leafnet":       {"net": "leafnet"},                                  # your own net
    "leafnet_gap":   {"net": "leafnet", "gap": True},                     # GAP, not flatten
    "leafnet_drop":  {"net": "leafnet", "dropout": 0.5},
    "leafnet_aug":   {"net": "leafnet", "geom": True, "colour": True},
    "plus_history":  {"data": "group+history"},                           # 807 images
    "history_only":  {"data": "history"},                                 # 700, other phones
}

# the combinations worth trying: the winning change plus one more each time
VARIANTS["aug_best"] = {"geom": True, "colour": True, "best_epoch": True}
VARIANTS["aug_long"] = {"geom": True, "colour": True, "epochs": 30}
VARIANTS["aug_cos_long"] = {"geom": True, "colour": True, "cosine": True, "epochs": 30}
VARIANTS["aug_data"] = {"geom": True, "colour": True, "data": "group+history"}

# "best" is the combination that won the rows above: augmentation, a cosine
# learning rate and more epochs, now on all the photographs we have. The two
# rows after it ask whether best-epoch weights and stronger dropout still add
# anything once those are in, and what augmentation is worth at the larger size.
VARIANTS["best"] = {"geom": True, "colour": True, "cosine": True, "epochs": 30,
                    "data": "group+history"}
VARIANTS["best_be"] = {"geom": True, "colour": True, "cosine": True, "epochs": 30,
                       "data": "group+history", "best_epoch": True}
VARIANTS["best_drop"] = {"geom": True, "colour": True, "cosine": True, "epochs": 30,
                         "data": "group+history", "dropout": 0.5}
VARIANTS["data_long"] = {"cosine": True, "epochs": 30, "data": "group+history"}


def settings_for(name):
    s = dict(BASE)
    for part in name.split(","):
        if part not in VARIANTS:
            raise SystemExit(f"unknown variant {part!r}; try --list")
        s.update(VARIANTS[part])
    s["name"] = name
    return s


# ------------------------------------------------------------ train, test


@torch.no_grad()
def accuracy(model, folder):
    where = next(model.parameters()).device      # the network decides, so a CPU model works too
    loader = DataLoader(LeafDataset(folder, base_transform()), batch_size=64)
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(where), labels.to(where)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset), loader.dataset


@torch.no_grad()
def confusion(model, folder):
    """Which species get mixed up — the matrix of cell 46, as text."""
    where = next(model.parameters()).device
    ds = LeafDataset(folder, base_transform())
    rows = torch.zeros(len(CLASSES), len(CLASSES), dtype=int)
    model.eval()
    for i in range(len(ds)):
        image, label = ds[i]
        pred = model(image.to(where).unsqueeze(0)).argmax(dim=1).item()
        rows[label][pred] += 1
    return rows


def run(s, seed):
    torch.manual_seed(seed)                  # same images, same order, same net
    model = make_leafnet(s) if s["net"] == "leafnet" else make_mobilenet(s)
    criterion = nn.CrossEntropyLoss(label_smoothing=s["smoothing"])
    params = [p for p in model.parameters() if p.requires_grad]
    if s["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=s["weight_decay"])
    else:
        optimizer = torch.optim.Adam(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s["epochs"]) \
        if s["cosine"] else None

    train_ds = train_dataset(s)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # the loop of cell 44, with three additions: the scheduler step, and the
    # option to remember the weights of the best epoch instead of the last one
    start, best_acc, best_weights = time.time(), -1.0, None
    for epoch in range(s["epochs"]):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        if scheduler is not None:
            scheduler.step()
        if s["best_epoch"]:
            acc, _ = accuracy(model, os.path.join(GROUP, "val"))
            if acc > best_acc:
                best_acc, best_weights = acc, copy.deepcopy(model.state_dict())
            print(f"  epoch {epoch + 1:2d}  loss {loss_sum / len(train_loader):.4f}"
                  f"  val {acc:.3f}  best so far {best_acc:.3f}", flush=True)
        else:
            print(f"  epoch {epoch + 1:2d}  loss {loss_sum / len(train_loader):.4f}", flush=True)
    if best_weights is not None:
        model.load_state_dict(best_weights)
    seconds = time.time() - start

    # three test sets: your 29, the camera-like 232, last year's 120
    val, val_ds = accuracy(model, os.path.join(GROUP, "val"))
    cam, cam_ds = accuracy(model, CAMVAL)
    seen_history = s["data"] != "group"          # never score on images it was trained on
    hist = None if seen_history else accuracy(model, os.path.join(HISTORY, "val"))[0]

    print(f"  {s['name']} seed {seed}: train {len(train_ds)}  "
          f"val {val:.3f} ({len(val_ds)})  camera-like {cam:.3f} ({len(cam_ds)})  "
          + (f"last year {hist:.3f} (120)" if hist is not None else
             "last year  -  (in the training data)")
          + f"  [{seconds:.0f}s]")
    tests = [("camera-like", CAMVAL)]
    if hist is not None:
        tests.append(("last year", os.path.join(HISTORY, "val")))
    for label, folder in tests:
        print(f"  confusion on {label} (rows = true, cols = predicted)")
        print("      " + " ".join(f"{c[:6]:>7}" for c in CLASSES))
        rows = confusion(model, folder)
        for cls, row in zip(CLASSES, rows):
            print(f"  {cls[:6]:>6} " + " ".join(f"{int(v):7d}" for v in row))

    # the weights of every run stay on disk (train/weights, not in git): the
    # winner can then be exported to ONNX by export_best.py without training again
    os.makedirs(WEIGHTS, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(WEIGHTS, f"{s['name'].replace(',', '+')}_seed{seed}.pt"))

    return dict(variant=s["name"], seed=seed, net=s["net"], train=len(train_ds),
                epochs=s["epochs"], val29=f"{val:.4f}", cam232=f"{cam:.4f}",
                hist120="" if hist is None else f"{hist:.4f}", secs=f"{seconds:.0f}",
                change=describe(s))


CHANGES = {"net": "network", "data": "training images", "epochs": "epochs",
           "geom": "crop/flip/rotation", "colour": "ColorJitter",
           "freeze": "frozen backbone", "dropout": "dropout",
           "gap": "global average pooling", "cosine": "cosine LR",
           "optimizer": "optimizer", "weight_decay": "weight decay",
           "smoothing": "label smoothing", "best_epoch": "best-epoch weights"}


def describe(s):
    """What this row changed against the baseline, in words."""
    return "; ".join(f"{name}={s[key]}" for key, name in CHANGES.items() if s[key] != BASE[key])


COLUMNS = ["variant", "seed", "net", "train", "epochs", "val29", "cam232", "hist120",
           "secs", "change"]


def append_result(row):
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def summary():
    """Mean and spread over the seeds, per variant, from results.csv."""
    rows = list(csv.DictReader(open(RESULTS)))
    names = []                                  # in the order they were first run
    for r in rows:
        if r["variant"] not in names:
            names.append(r["variant"])
    print(f"{'variant':14}{'runs':>5}{'train':>7}  "
          f"{'your val (29)':>16}{'camera-like (232)':>19}{'last year (120)':>17}   change")
    for name in names:
        runs = [r for r in rows if r["variant"] == name]
        line = f"{name:14}{len(runs):>5}{runs[0]['train']:>7}"
        for column, width in (("val29", 16), ("cam232", 19), ("hist120", 17)):
            values = [float(r[column]) for r in runs if r[column]]
            if not values:
                line += f"{'- (in training)':>{width}}" if column == "hist120" \
                    else f"{'-':>{width}}"
            else:
                mean_acc = sum(values) / len(values)
                spread = (sum((v - mean_acc) ** 2 for v in values)
                          / max(len(values) - 1, 1)) ** 0.5
                line += f"{mean_acc:.3f} +-{spread:.3f}".rjust(width)
        print(line + f"   {runs[0]['change'] or 'the Task 2 model, unchanged'}")
    print("\none image on the 29-image set is 3.4 %, so read differences there as a guide only;"
          "\nthe 232 and 120 image sets are the ones to trust, +- is the spread over the seeds.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="baseline",
                   help="one name from --list, a comma-separated combination, or 'all'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="", help="e.g. 0,1,2 — repeats the run with each seed")
    p.add_argument("--epochs", type=int, default=0,
                   help="override the number of epochs, e.g. 2 for a quick try")
    p.add_argument("--summary", action="store_true", help="print the table and stop")
    p.add_argument("--list", action="store_true", help="print the variant names and stop")
    a = p.parse_args()

    if a.list:
        for name, change in VARIANTS.items():
            print(f"{name:14} {change if change else '(nothing: the Task 2 model)'}")
        return
    if a.summary:
        summary()
        return

    names = list(VARIANTS) if a.variant == "all" else a.variant.split(",")
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [a.seed]
    if not os.path.isdir(CAMVAL):
        raise SystemExit("data/camval is missing — run train/make_camera_val.py once first")

    for name in names:
        s = settings_for(name)
        if a.epochs:
            s["epochs"] = a.epochs
        print(f"\n=== {name}: {describe(s) or 'baseline'} on {device}", flush=True)
        for seed in seeds:
            append_result(run(s, seed))
    print()
    summary()


if __name__ == "__main__":
    main()
