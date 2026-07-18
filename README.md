# Sine-Wave-Estimation

A small PyTorch project that fits three models &mdash; a single-layer **Linear** regressor, a 2-layer **MLP**, and an **LSTM** &mdash; to a synthetic sine wave and compares how well each one learns to predict the next value of the sequence. The data spans two full cycles over `[0, 4π]`, so the models are evaluated on a genuinely periodic signal rather than a single arc.

## Overview

`main.py` generates a sine wave `sin(x)` sampled every `step_x` radians, slices it into overlapping input windows with the `slide()` helper in `util.py`, splits the windows into train / validation / test sets, and trains all three models with the same shared loop (Adam + MSE loss, mini-batches). After training it writes a set of figures to `assets/` and prints the test MSE for each model.

The point of the comparison is to show how model capacity tracks the difficulty of the task: the **Linear** model can only approximate a tangent at best, the **MLP** bends toward the curve, and the **LSTM** &mdash; built for sequences &mdash; tracks the wave closely across both cycles.

## Results

Test MSE over the `[0, 4π]` range (lower is better):

| Model | Test MSE |
| --- | --- |
| Linear (1-layer) | 0.000313 |
| MLP (2-layer) | 0.000299 |
| LSTM | 0.000076 |

The **LSTM** benefits most from the longer sequence and reaches the lowest error of the three.

## Project structure

```
Sine-Wave-Estimation/
├── main.py        # training + evaluation pipeline; writes figures to assets/
├── config.py      # all hyperparameters and data settings
├── util.py        # data-preprocessing utilities (slide())
├── assets/        # generated figures (git-ignored; produced by main.py)
└── .gitignore
```

There is no `src/` package and no separate article tooling &mdash; the three files above are the whole project.

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

Install them with pip:

```bash
pip install torch numpy matplotlib
```

## Configuration

Every tunable value lives in `config.py` &mdash; nothing is hard-coded inside `main.py`. Edit a single constant to change behaviour.

| Constant | Default | Meaning |
| --- | --- | --- |
| `step_x` | `0.05` | Sampling step for the sine wave (radians) |
| `look_back` | `15` | Number of past values used as input to each window |
| `hope` | `2` | Stride between consecutive sliding windows |
| `n_epoch` | `1000` | Number of training epochs |
| `n_batch` | `32` | Mini-batch size |
| `lr` | `0.01` | Learning rate for the Adam optimizer (all models) |
| `trn_sz` | `0.6` | Training-set proportion |
| `vld_sz` | `0.1` | Validation-set proportion (remainder is test) |
| `vld_step` | `50` | Evaluate validation loss every N epochs |
| `hidden_sz` | `20` | Neurons in the MLP hidden layer |
| `lstm_hidden_sz` | `20` | Units in the LSTM hidden state |
| `lstm_n_layer` | `1` | Number of stacked LSTM layers |

## Usage

From the repository root:

```bash
python main.py
```

The script prints per-model progress and test MSE, then writes a set of figures into `assets/`.

## Generated figures

Running `main.py` produces one figure per concept, per model, saved as JPGs in `assets/`. For each of the three models you get:

- a **learning curves** figure (training vs. validation loss per epoch),
- a **predictions** figure (model output vs. ground truth over the full `[0, 4π]` range),
- **epoch snapshots** &mdash; the model's output after the 1st and 5th epochs, showing how the fit develops during early training,
- an **output progression** figure that overlays epoch 1, epoch 5, and the final fit on the same axes.

A final **model comparison** figure overlays all three trained models on the ground-truth wave. The x-axis is labelled in multiples of π (from `0` to `4π`) and the y-axis spans `[-1.1, 1.1]`.

Note that the `assets/` folder is listed in `.gitignore`, so the figures are produced locally each time you run the script and are not checked into version control.

## Attribution

Full attribution and license information for every library this project depends on is in [`RESOURCES.md`](RESOURCES.md).