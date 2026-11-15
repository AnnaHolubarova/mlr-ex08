# MLR exercise 8 — leaf classifier on your phone

**The page: <https://annaholubarova.github.io/mlr-ex08/>**
(a QR code of the same URL sits at the bottom of the page, so a phone can open it straight from
a screen or from this README.)

Open the link on a phone, tap **Start camera**, allow the camera, and point it at a leaf. The
MobileNet from the exercise-8 notebook runs entirely in the browser — the photograph is not
uploaded anywhere. On a desktop, or if the camera is refused, **Upload a photo instead** does the
same thing with a still image.

| file | what it is |
| --- | --- |
| `index.html` | the whole app: camera, preprocessing, inference, results, self-check, QR code |
| `leaf_model.onnx` | the trained MobileNetV3-Small, exported from the notebook (opset `ai.onnx` 20, 6.3 MB) |
| `model_info.json` | what the export says about the model: 4 classes, `input_size` 96, mean, std |
| `check_image.jpg` | a 96 × 96 validation image shipped with the model |
| `check_expected.json` | the probabilities the notebook computed for that image |
| `make_qr.py`, `qr.svg` | how the QR code was made (offline, `segno`) and its standalone copy |

The four files that came out of `ex08_for_agent.zip` — `leaf_model.onnx`, `model_info.json`,
`check_image.jpg`, `check_expected.json` — are byte-for-byte what the notebook produced, unchanged.

## How the page works

`model_info.json` is read at run time and drives everything, so the page cannot drift away from the
model: it takes the class names, the input size and the mean/std from there rather than hard-coding
them. Each frame goes through the same five steps the notebook's
`transforms.Compose([ToTensor(), Normalize(mean, std)])` describes, and each is marked with a
`STEP` comment in `index.html`:

1. **crop** the central square of the frame,
2. **resize** it to 96 × 96,
3. **scale** the RGB bytes by 1/255 → 0…1 (= `ToTensor`),
4. **standardize** each channel with `(v − mean[c]) / std[c]` (= `Normalize`),
5. **lay it out channels-first**, planar R, G, B → one `float32` tensor of `1 × 3 × 96 × 96`.

The model ends in raw logits (`CrossEntropyLoss` folded the softmax into training), so the page
applies softmax itself, exactly as the notebook's `model(x).softmax(dim=1)` does.

Inference is ONNX Runtime Web running the network as WebAssembly in the tab, on one thread —
GitHub Pages cannot serve the cross-origin-isolation headers that `SharedArrayBuffer` needs.
Runtime Web is pinned to `1.31.1` from jsDelivr; older builds are not an option because this model
was exported at opset 20, which `1.14.0` and `1.17.3` reject. WebAssembly needs SIMD, so iPhone
wants iOS 16.4 or newer; anything older still gets the photo upload.

## Nothing leaves the phone

The page makes two kinds of request, both visible in the Network panel:

* its own files from this GitHub Pages site (`leaf_model.onnx`, `model_info.json`,
  `check_expected.json`, `check_image.jpg`),
* ONNX Runtime Web and its `.wasm` from jsDelivr — a pinned, fixed library file, the same for every
  visitor, with nothing from the camera in it. If jsDelivr fails to answer, the runtime is retried
  from `unpkg.com` (the same pinned `1.31.1` build, `ORT_BUILDS` in `index.html`), so a second CDN
  origin can appear in the log — measured runs never needed it.

No image, frame, filename or identifier is ever sent anywhere: the pixel data goes from
`<video>` into a canvas and from there into a `Float32Array` in this tab. No fonts, no analytics,
no cookies, no external QR-image service (the QR code is an SVG drawn into `index.html`).

## Checking it

* **Same numbers as the notebook** — tap **Self-check: check_image.jpg** (or open the page with
  `?selftest=1`). It pushes `check_image.jpg` through the identical camera pipeline and prints its
  probabilities next to `check_expected.json`, with the difference per class. Measured in Chrome:
  every class agrees to six decimals (largest difference 0.000000), top class *cherry*.
  **Re-checking it in Python needs a `libjpeg-turbo` decoder.** Decoding `check_image.jpg` with
  OpenCV or `torchvision.io` and running `leaf_model.onnx` on those pixels reproduces
  `check_expected.json` to 1.3e-07 — pixel-for-pixel the same bytes Chrome's canvas produces.
  Decoding the same file with Pillow produces different pixels (26 % of them, by up to 16 levels)
  and lands 0.0017 off on the top class, which looks like a broken page but is the JPEG decoder:
  this model is certain of its answer, so a one-level pixel change moves a class sitting at 0.0005
  by a quarter of its value. Compare through the page, or through a `libjpeg-turbo` decode.
* **What the model is fed** — the "What the model is fed" box at the bottom of the page restates
  the classes, the tensor shape and the pipeline that `model_info.json` implies.
* **The QR code** — `python3 -I make_qr.py AnnaHolubarova mlr-ex08` regenerates it from the Pages
  URL, so the encoded address can be re-derived, not just trusted.

## Task 4 — measuring what each change brings

`train/` is the measuring: the Task 3 pipeline with one switch on top, a fixed baseline, one change
per row, the same training images, a fixed seed repeated, and every row scored on the same three
test sets. `train/README.md` says how to run it; `train/results.csv` is one line per run.

| file | what it is |
| --- | --- |
| `train/variants.py` | the baseline and every variant; `--summary` prints the table below |
| `train/make_camera_val.py` | the camera-like test set: the 29 validation leaves cropped, rotated and recoloured once, with a fixed seed |
| `train/results.csv` | 91 runs — variant, seed, the three accuracies, and what that row changed |
| `train/grid.log` | the epoch output and the confusion matrices behind those rows |
| `train/export_best.py` | exports the chosen variant to ONNX the way task 2e did, and checks it against PyTorch |
| `leaf_model_best.onnx`, `model_info_best.json` | the chosen variant, for the course camera app |

The baseline is the model this page runs. Mean over 3–5 seeds:

| variant | camera-like (232) | last year's photos (120) |
| --- | --- | --- |
| baseline — the model on this page | 0.566 ±0.111 | 0.478 ±0.074 |
| + ColorJitter (colour) | 0.652 ±0.048 | 0.535 ±0.051 |
| + cropping, flipping, rotation | 0.562 ±0.060 | 0.447 ±0.065 |
| + both (= augmentation) | 0.680 ±0.034 | 0.592 ±0.037 |
| + cosine learning rate and 30 epochs | 0.728 ±0.024 | 0.627 ±0.028 |
| + last year's 700 photographs → `leaf_model_best.onnx` | 0.766 ±0.017 | — trained on them |

What did **not** help on the camera-like set, all inside the spread of the baseline: AdamW with
weight decay 0.526, training 30 epochs instead of 15 without augmentation 0.516, label smoothing
0.580, dropout 0.2 → 0.5 0.585, keeping the best epoch's weights 0.598. Freezing the pretrained
backbone looked fine on the camera-like set (0.644) but fell to 0.422 on last year's photographs.
In the small from-scratch network of Task 3, global average pooling (0.348) and dropout (0.398)
moved it only a little over its 0.332, and augmentation brought 0.460 — still far below the
pretrained network. Choosing the epoch on the 29-image validation set raised that set to 0.828 while
the camera-like number went *down* to 0.700: it optimised the wrong number.

The export was checked the way task 2f was: PyTorch and the exported file differ by 1.2e-06 on the
same photographs, and the page pipeline of this repository, run on a local copy with this model,
reproduced its probabilities to 0.00007 and classified the fake camera stream in 6.5 ms a frame.
The deployed page deliberately keeps `leaf_model.onnx`, the model task 2f was verified with.

## Course context

Machine Learning in Research (MLR), exercise 8. Task 2f: a model trained and exported in the course
notebook, put on a phone camera by a coding agent, and then checked by hand — `REVIEW.md` is the
student's own write-up of those checks. Task 4: improving that model one change at a time —
`REPORT.md` is the student's own reading of the numbers above.
