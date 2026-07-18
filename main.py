# main.py - full training and evaluation pipeline (PyTorch only)

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from config import *
from util import slide

torch.manual_seed(42)
np.random.seed(42)

# 1. Generate sine wave data over [0, 4*pi] (two full cycles)
seq = np.sin(np.arange(0, 4 * np.pi, step_x))
x, y = slide(seq, look_back, hope)

# 2. Split into train, validation, and test sets
n = len(x)
trn_end = int(n * trn_sz)
vld_end = int(n * (trn_sz + vld_sz))

x_trn, y_trn = x[:trn_end], y[:trn_end]
x_vld, y_vld = x[trn_end:vld_end], y[trn_end:vld_end]
x_tst, y_tst = x[vld_end:], y[vld_end:]

# Flatten targets to 1D arrays
y_trn = y_trn.ravel()
y_vld = y_vld.ravel()
y_tst = y_tst.ravel()


# 3. PyTorch model definitions
class LinearModel(nn.Module):
    """Single linear layer: y = Wx + b (linear regression via gradient descent)."""

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class MLPModel(nn.Module):
    """Two-layer MLP: Linear -> ReLU -> Linear."""

    def __init__(self, input_dim, hidden_sz):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_sz)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sz, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x))).squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM recurrent network with a linear regression head."""

    def __init__(self, input_dim=1, hidden_sz=20, n_layer=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_sz, num_layers=n_layer, batch_first=True)
        self.fc = nn.Linear(hidden_sz, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]           # take last time step
        return self.fc(out).squeeze(-1)


def to_tensor(arr, lstm=False):
    """Convert numpy arrays to torch tensors. LSTM expects 3D (samples, look_back, 1)."""
    if lstm:
        t = torch.from_numpy(arr.reshape(arr.shape[0], arr.shape[1], 1)).float()
    else:
        t = torch.from_numpy(arr).float()
    return t


def train_model(model, x_trn, y_trn, x_vld, y_vld, x_full, epochs, lr, batch,
                snap_epochs):
    """Shared training loop: Adam + MSELoss, mini-batches, snapshots.

    Snapshots and final predictions are produced over the FULL input set (x_full)
    so the entire sine wave (0..4*pi) is visible in figures.

    Returns:
        trn_losses (list), vld_losses (list), snap_preds (dict {epoch: array}),
        full_pred (np.array over all windows)
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    y_trn_t = torch.from_numpy(y_trn).float()
    y_vld_t = torch.from_numpy(y_vld).float()

    trn_losses, vld_losses = [], []
    snap_preds = {}
    n_samples = x_trn.shape[0]

    for epoch in range(epochs):
        # ---- Mini-batch training ----
        perm = torch.randperm(n_samples)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n_samples, batch):
            idx = perm[start:start + batch]
            xb = x_trn[idx]
            yb = y_trn_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        trn_losses.append(epoch_loss / max(n_batches, 1))

        # ---- Validation ----
        if epoch % vld_step == 0 or epoch == epochs - 1:
            with torch.no_grad():
                vld_pred = model(x_vld)
                vld_mse = loss_fn(vld_pred, y_vld_t).item()
            vld_losses.append(vld_mse)
        else:
            vld_losses.append(vld_losses[-1] if vld_losses else 0.0)

        # ---- Snapshots over the FULL input set ----
        if (epoch + 1) in snap_epochs:
            with torch.no_grad():
                snap_preds[epoch + 1] = model(x_full).numpy()

    # ---- Final predictions over the FULL input set ----
    with torch.no_grad():
        full_pred = model(x_full).numpy()

    return trn_losses, vld_losses, snap_preds, full_pred


# 4. Prepare tensor inputs for each model family
x_trn_lin = to_tensor(x_trn)
x_vld_lin = to_tensor(x_vld)
x_tst_lin = to_tensor(x_tst)
x_full_lin = to_tensor(x)

x_trn_lstm = to_tensor(x_trn, lstm=True)
x_vld_lstm = to_tensor(x_vld, lstm=True)
x_tst_lstm = to_tensor(x_tst, lstm=True)
x_full_lstm = to_tensor(x, lstm=True)

snap_epochs = [1, 5]

# 5. Train all three models
print("Training Linear model...")
linear_model = LinearModel(look_back)
lin_trn, lin_vld, lin_snap, linear_full_pred = train_model(
    linear_model, x_trn_lin, y_trn, x_vld_lin, y_vld, x_full_lin,
    n_epoch, lr, n_batch, snap_epochs)
linear_tst_pred = linear_full_pred[vld_end:]
linear_tst_mse = np.mean((linear_tst_pred - y_tst) ** 2)
print(f"Linear Model -- Test MSE: {linear_tst_mse:.6f}")

print("\nTraining 2-layer MLP...")
mlp_model = MLPModel(look_back, hidden_sz)
mlp_trn, mlp_vld, mlp_snap, mlp_full_pred = train_model(
    mlp_model, x_trn_lin, y_trn, x_vld_lin, y_vld, x_full_lin,
    n_epoch, lr, n_batch, snap_epochs)
mlp_tst_pred = mlp_full_pred[vld_end:]
mlp_tst_mse = np.mean((mlp_tst_pred - y_tst) ** 2)
print(f"2-layer MLP -- Test MSE: {mlp_tst_mse:.6f}")

print("\nTraining LSTM...")
lstm_model = LSTMModel(input_dim=1, hidden_sz=lstm_hidden_sz, n_layer=lstm_n_layer)
lstm_trn, lstm_vld, lstm_snap, lstm_full_pred = train_model(
    lstm_model, x_trn_lstm, y_trn, x_vld_lstm, y_vld, x_full_lstm,
    n_epoch, lr, n_batch, snap_epochs)
lstm_tst_pred = lstm_full_pred[vld_end:]
lstm_tst_mse = np.mean((lstm_tst_pred - y_tst) ** 2)
print(f"LSTM         -- Test MSE: {lstm_tst_mse:.6f}")


# 6. Plotting: save all figures (separately) into assets/ as JPG (quality=30)
os.makedirs('assets', exist_ok=True)
jpg_kwargs = {'quality': 30}

# Map each window (across the FULL dataset) to its x-position in radians.
# Window j has target = seq[j*hope + look_back], whose radian position is
# (j*hope + look_back) * step_x.
x_radians = (np.arange(len(y)) * hope + look_back) * step_x

# Tick labels in multiples of pi for the x-axis (0 to 4*pi)
pi_ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi,
            5 * np.pi / 2, 3 * np.pi, 7 * np.pi / 2, 4 * np.pi]
pi_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$',
             r'$5\pi/2$', r'$3\pi$', r'$7\pi/2$', r'$4\pi$']

# High-contrast color palette
C_TRUTH = '#000000'   # black  - ground truth
C_TRAIN = '#D62728'   # red    - training loss
C_VLD   = '#1F77B4'   # blue   - validation loss
C_LIN   = '#FF7F0E'   # orange - linear model
C_MLP   = '#2CA02C'   # green  - MLP model
C_LSTM  = '#17BECF'   # cyan   - LSTM model
C_SNAP1 = '#9467BD'   # purple - snapshot (epoch 1)
C_SNAP5 = '#8C564B'   # brown  - snapshot (epoch 5)


def _style_sine_ax(ax):
    """Apply common styling: x-axis in radians, y-axis in [-1.1, 1.1]."""
    ax.set_xticks(pi_ticks)
    ax.set_xticklabels(pi_labels)
    ax.set_xlim(0, 4 * np.pi)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('x (radians)')
    ax.grid(alpha=0.3)


def save_predictions(path, title, pred, label, color):
    """Save a single predictions-vs-ground-truth figure over 0..4*pi."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_radians, y, label='Ground Truth', linestyle='--',
            linewidth=2, color=C_TRUTH)
    ax.plot(x_radians, pred, label=label, linestyle=':', linewidth=2.5,
            color=color)
    _style_sine_ax(ax)
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, pil_kwargs=jpg_kwargs)
    plt.close()


def save_learning_curves(path, title, trn_losses, vld_losses):
    """Save a learning-curves figure."""
    plt.figure(figsize=(7, 5))
    plt.plot(trn_losses, label='Training Loss', linestyle='--', linewidth=1.5,
             color=C_TRAIN)
    plt.plot(vld_losses, label='Validation Loss', linestyle=':', linewidth=1.5,
             color=C_VLD)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, pil_kwargs=jpg_kwargs)
    plt.close()


def save_snapshots(path, title, snap_preds, final_pred, final_color, final_label):
    """Save a progression figure: epoch 1 vs epoch 5 vs final (over 0..4*pi)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_radians, y, label='Ground Truth', linestyle='--',
            linewidth=2, color=C_TRUTH)
    if 1 in snap_preds:
        ax.plot(x_radians, snap_preds[1], label='Epoch 1', linestyle=':',
                linewidth=2.5, color=C_SNAP1)
    if 5 in snap_preds:
        ax.plot(x_radians, snap_preds[5], label='Epoch 5', linestyle='-.',
                linewidth=2.5, color=C_SNAP5)
    ax.plot(x_radians, final_pred, label=final_label, linestyle=':',
            linewidth=2.5, color=final_color)
    _style_sine_ax(ax)
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, pil_kwargs=jpg_kwargs)
    plt.close()


# Linear model figures
save_learning_curves('assets/linear_learning_curves.jpg', 'Learning Curves (Linear)',
                     lin_trn, lin_vld)
save_predictions('assets/predictions_linear.jpg', 'Test Set Predictions - Linear (1-layer)',
                 linear_full_pred, 'Linear (1-layer)', C_LIN)
if 1 in lin_snap:
    save_predictions('assets/linear_output_epoch1.jpg', 'Linear Output After 1st Epoch',
                     lin_snap[1], 'Linear after epoch 1', C_SNAP1)
if 5 in lin_snap:
    save_predictions('assets/linear_output_epoch5.jpg', 'Linear Output After 5th Epoch',
                     lin_snap[5], 'Linear after epoch 5', C_SNAP5)
save_snapshots('assets/linear_output_progression.jpg', 'Linear Output Progression',
               lin_snap, linear_full_pred, C_LIN, 'Final')

# MLP figures
save_learning_curves('assets/learning_curves.jpg', 'Learning Curves (2-layer MLP)',
                     mlp_trn, mlp_vld)
save_predictions('assets/predictions_mlp.jpg', 'Test Set Predictions - MLP (2-layer)',
                 mlp_full_pred, 'MLP (2-layer)', C_MLP)
if 1 in mlp_snap:
    save_predictions('assets/mlp_output_epoch1.jpg', 'MLP Output After 1st Epoch',
                     mlp_snap[1], 'MLP after epoch 1', C_SNAP1)
if 5 in mlp_snap:
    save_predictions('assets/mlp_output_epoch5.jpg', 'MLP Output After 5th Epoch',
                     mlp_snap[5], 'MLP after epoch 5', C_SNAP5)
save_snapshots('assets/mlp_output_progression.jpg', 'MLP Output Progression',
               mlp_snap, mlp_full_pred, C_MLP, 'Final')

# LSTM figures
save_learning_curves('assets/lstm_learning_curves.jpg', 'Learning Curves (LSTM)',
                     lstm_trn, lstm_vld)
save_predictions('assets/predictions_lstm.jpg', 'Test Set Predictions - LSTM',
                 lstm_full_pred, 'LSTM', C_LSTM)
if 1 in lstm_snap:
    save_predictions('assets/lstm_output_epoch1.jpg', 'LSTM Output After 1st Epoch',
                     lstm_snap[1], 'LSTM after epoch 1', C_SNAP1)
if 5 in lstm_snap:
    save_predictions('assets/lstm_output_epoch5.jpg', 'LSTM Output After 5th Epoch',
                     lstm_snap[5], 'LSTM after epoch 5', C_SNAP5)
save_snapshots('assets/lstm_output_progression.jpg', 'LSTM Output Progression',
               lstm_snap, lstm_full_pred, C_LSTM, 'Final')

# Model comparison figure (over the full 0..4*pi range)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_radians, y, label='Ground Truth', linestyle='--', linewidth=2,
        color=C_TRUTH)
ax.plot(x_radians, linear_full_pred, label='Linear', linestyle=':', linewidth=2.5,
        color=C_LIN)
ax.plot(x_radians, mlp_full_pred, label='MLP', linestyle='-.', linewidth=2.5,
        color=C_MLP)
ax.plot(x_radians, lstm_full_pred, label='LSTM', linestyle=':', linewidth=2.5,
        color=C_LSTM)
_style_sine_ax(ax)
ax.set_ylabel('y')
ax.set_title('Model Comparison - Predictions over [0, 4π]')
ax.legend()
plt.tight_layout()
plt.savefig('assets/model_comparison.jpg', dpi=150, pil_kwargs=jpg_kwargs)
plt.close()

print("\nFigures saved to assets/ (Linear, MLP, LSTM learning curves, predictions, "
      "epoch snapshots, progression, and model comparison).")

# 7. Final results summary
print("\n===== FINAL RESULTS (data range: 0 to 4*pi) =====")
print(f"Linear Model  Test MSE: {linear_tst_mse:.6f}")
print(f"2-layer MLP   Test MSE: {mlp_tst_mse:.6f}")
print(f"LSTM          Test MSE: {lstm_tst_mse:.6f}")