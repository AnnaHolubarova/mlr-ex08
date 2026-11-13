"""Export the chosen model exactly the way 2e exported the first one.

    python3 train/export_best.py --variant best --seed 0

It rebuilds that variant's network, loads the weights train/variants.py saved
for it, and writes leaf_model_best.onnx in the repository folder with the same
`model_info` metadata inside the file, plus model_info_best.json beside it —
the same two things the course camera app and the page need. The exported file
and PyTorch are then run on the same photographs and their probabilities
printed side by side, so you can see the export did not change the model.

The page of Task 2 keeps the leaf_model.onnx it was deployed with; this script
never touches it, it only writes the *_best files.
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from variants import (CLASSES, GROUP, HISTORY, CAMVAL, LeafDataset, RESULTS, WEIGHTS,  # noqa: E402
                      accuracy, base_transform, make_leafnet, make_mobilenet, settings_for)

ROOT = os.path.join(HERE, "..")
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]      # cell 40


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="best")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    weights = os.path.join(WEIGHTS, f"{a.variant.replace(',', '+')}_seed{a.seed}.pt")
    if not os.path.exists(weights):
        raise SystemExit(f"no saved weights {weights} — train it first: "
                         f"python3 train/variants.py --variant {a.variant} --seed {a.seed}")

    s = settings_for(a.variant)
    model = make_leafnet(s) if s["net"] == "leafnet" else make_mobilenet(s)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval().cpu()                                  # cell 49

    out = os.path.join(ROOT, "leaf_model_best.onnx")
    torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), out,
                      input_names=["image"], output_names=["logits"], external_data=False)

    model_info = {"classes": CLASSES, "input_size": 96, "mean": mean, "std": std}
    import onnx
    onnx_model = onnx.load(out)
    onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
    onnx.save(onnx_model, out)
    with open(os.path.join(ROOT, "model_info_best.json"), "w") as f:
        json.dump(model_info, f)
    print(f"{os.path.basename(out)}: {os.path.getsize(out) / 1e6:.1f} MB, "
          f"opset {onnx_model.opset_import[0].version}, metadata {model_info}")

    # the same photographs through PyTorch and through the ONNX file
    import onnxruntime
    session = onnxruntime.InferenceSession(out, providers=["CPUExecutionProvider"])
    transform = base_transform()
    worst = 0.0
    for path in [os.path.join(GROUP, "val", "apple", sorted(os.listdir(os.path.join(GROUP, "val", "apple")))[0]),
                 os.path.join(HISTORY, "val", "maple", sorted(os.listdir(os.path.join(HISTORY, "val", "maple")))[0])]:
        x = transform(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            pytorch = torch.softmax(model(x), dim=1)[0]
        logits = torch.tensor(session.run(["logits"], {"image": x.numpy()})[0])   # (1, 4)
        onnx_probs = torch.softmax(logits, dim=1)[0]
        worst = max(worst, float((pytorch - onnx_probs).abs().max()))
        print(f"\n{os.path.relpath(path, ROOT)}")
        for cls, pytorch_p, onnx_p in zip(CLASSES, pytorch.tolist(), onnx_probs.tolist()):
            print(f"  {cls:9} torch {pytorch_p:.4f}   onnx {onnx_p:.4f}")
    print(f"\nlargest difference between torch and onnx: {worst:.1e}")

    # what this variant measured, straight out of results.csv
    import csv as csv_module
    rows = [r for r in csv_module.DictReader(open(RESULTS)) if r["variant"] == a.variant]
    if rows:
        print(f"\nmeasured in the grid as {a.variant}: " + "; ".join(
            f"seed {r['seed']} val {float(r['val29']):.3f} camera-like {float(r['cam232']):.3f}"
            + (f" last year {float(r['hist120']):.3f}" if r["hist120"] else "") for r in rows))
    # only the sets this variant has not been trained on count as a measurement
    print(f"and measured again from the exported weights: "
          f"val {accuracy(model, os.path.join(GROUP, 'val'))[0]:.3f}  "
          f"camera-like {accuracy(model, CAMVAL)[0]:.3f}"
          + ("  (last year's photos were part of its training data, so they are not a test)"
             if s["data"] != "group" else
             f"  last year {accuracy(model, os.path.join(HISTORY, 'val'))[0]:.3f}"))
    print("\nthis file is for the course camera app; the deployed page still uses leaf_model.onnx")


if __name__ == "__main__":
    main()
