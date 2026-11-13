"""Make the camera-like validation set ONCE, then never touch it again.

The validation folder of the notebook holds 29 photographs taken in the same
session as the training ones, with the same phones and roughly the same light.
A model can score well on that and still fail on the phone camera, which shows
a new tree, a new student, a new phone, a tilted leaf and a shadow. So every
variant is also measured on a set that looks like a camera:

    the 29 validation photographs, each copied 8 times and then
    randomly cropped, rotated, flipped and recoloured with one fixed seed

232 images, written to data/camval as ordinary JPEGs so you can open the
folder and look at what the network is being asked. The seed is fixed here
(SEED = 7) and has nothing to do with the training seed, so every run of every
variant is scored on exactly the same pictures.

    python3 train/make_camera_val.py          # run once
"""
import os
from PIL import Image
from torchvision import transforms

SEED = 7                          # fixed, once and for all
COPIES = 8                        # distorted versions made of every image
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data", "leaves", "val")
OUT = os.path.join(HERE, "..", "data", "camval")

# crop, rotate, recolour: the three things a hand-held phone does to a leaf
camera_like = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.45, 1.0)),   # leaf fills less of the frame
    transforms.RandomRotation(30),                         # the leaf is not upright
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
])

import torch     # imported after SEED so the order of seeding stays obvious


def main():
    classes = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
    made = 0
    for cls in classes:
        os.makedirs(os.path.join(OUT, cls), exist_ok=True)
        for name in sorted(os.listdir(os.path.join(SRC, cls))):
            image = Image.open(os.path.join(SRC, cls, name)).convert("RGB")
            for copy in range(COPIES):
                torch.manual_seed(SEED * 1000 + made)      # same pictures for every run
                out = camera_like(image)
                out.save(os.path.join(OUT, cls, f"{name[:-4]}_{copy}.jpg"), quality=95)
                made += 1
    print(f"wrote {made} images to {os.path.relpath(OUT, os.path.join(HERE, '..'))}")
    print("this is a fixed test set: make it once, do not regenerate it between runs")


if __name__ == "__main__":
    main()
