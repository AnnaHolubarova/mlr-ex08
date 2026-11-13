# Task 4 — measuring what each change brings

Three small files, and they are the Task 3 pipeline with a switch on top: the
`LeafDataset` you wrote, the `LeafNet` you wrote, the training loop of Task 2.
Nothing was rewritten and nothing was added that a variant does not need, so
every line in the results can be pointed at in the report.

| file | what it does |
| --- | --- |
| `make_camera_val.py` | makes the camera-like test set **once** (232 JPEGs in `data/camval`) |
| `variants.py` | trains one variant with one seed, appends a row to `results.csv` |
| `export_best.py` | exports the chosen variant to `leaf_model_best.onnx`, the way 2e did |
| `results.csv` | one row per run: the numbers, the seed and what was changed |
| `grid.log` | the epoch-by-epoch output of the whole grid, for reading afterwards |

## Run it

From the repository root (the images go in `data/leaves` and
`data/leaves_history`, the same folders the notebook downloads; they are not
in git):

```bash
python3 train/make_camera_val.py                              # once, fixed seed 7
python3 train/variants.py --list                              # the names you can ask for
python3 train/variants.py --variant baseline --seed 0         # one run
python3 train/variants.py --variant aug --seeds 0,1,2         # three seeds
python3 train/variants.py --variant aug,cosine --seed 0       # a combination
python3 train/variants.py --variant all --seeds 0,1,2         # the whole grid
python3 train/variants.py --summary                           # the table, from the csv
python3 train/export_best.py --variant best --seed 0          # the file for the camera app
```

On a laptop with Apple Silicon it trains on the GPU (`mps`); elsewhere it falls
back to the CPU. On Colab it is the same code, with `device = "cpu"` and the
data downloaded by the notebook's own cells.

## What is kept fixed, so a row means one change

* the same 107 group training photographs and the same 29 validation ones,
  except in the two rows that say otherwise;
* `torch.manual_seed(seed)` before the network is built, so the weights and the
  batch order are the same for the same seed;
* the baseline everywhere: the pretrained `mobilenet_v3_small` of Task 2, Adam
  with lr 1e-3, cross-entropy, 15 epochs, batches of 32 — the model that is on
  the phone now;
* one change per row, listed in the last column of the table.

## Three test sets, because one number is not enough

| column | images | what it can see |
| --- | --- | --- |
| `val29` | 29 | the notebook's number — one image is 3.4 %, so it moves by itself |
| `cam232` | 232 | the same validation leaves, cropped, rotated and recoloured once with a fixed seed: what a hand-held phone shows |
| `hist120` | 120 | last year's photographs: another session, other students, other phones |

The rows that trained on last year's photographs have `hist120` empty on
purpose — a model is never scored on images it has seen.

The last column of the table is the spread `+-` over the seeds. On 29 images a
difference below about 0.10 means nothing; the two bigger sets are what a
conclusion should rest on, and a change is only worth keeping when it wins on
those and is larger than the spread.
