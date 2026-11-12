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
  visitor, with nothing from the camera in it.

No image, frame, filename or identifier is ever sent anywhere: the pixel data goes from
`<video>` into a canvas and from there into a `Float32Array` in this tab. No fonts, no analytics,
no cookies, no external QR-image service (the QR code is an SVG drawn into `index.html`).

## Checking it

* **Same numbers as the notebook** — tap **Self-check: check_image.jpg** (or open the page with
  `?selftest=1`). It pushes `check_image.jpg` through the identical camera pipeline and prints its
  probabilities next to `check_expected.json`, with the difference per class. Measured in Chrome:
  every class agrees to six decimals (largest difference 0.000000), top class *cherry*.
* **What the model is fed** — the "What the model is fed" box at the bottom of the page restates
  the classes, the tensor shape and the pipeline that `model_info.json` implies.
* **The QR code** — `python3 -I make_qr.py AnnaHolubarova mlr-ex08` regenerates it from the Pages
  URL, so the encoded address can be re-derived, not just trusted.

## Course context

Machine Learning in Research (MLR), exercise 8, task 2f: a model trained and exported in the course
notebook, put on a phone camera by a coding agent, and then checked by hand. `REVIEW.md` in this
repository is the student's own write-up of those checks.
