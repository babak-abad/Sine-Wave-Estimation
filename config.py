# config.py - all hyperparameters and data settings

# Data generation
step_x = 0.05         # Sampling step for sine wave (over [0, 2*pi])
look_back = 15        # Number of past values used as input
hope = 2              # Step size for sliding window (stride)

# Training hyperparameters
n_epoch = 1000        # Number of training epochs
n_batch = 32          # Batch size for training
lr = 0.01             # Learning rate for optimizer (Adam; used by all PyTorch models)

# Dataset split ratios
trn_sz = 0.6          # Training set proportion
vld_sz = 0.1          # Validation set proportion (remaining is test)

# Validation frequency
vld_step = 50         # Evaluate validation loss every N epochs

# Model architecture
hidden_sz = 20        # Number of neurons in the MLP hidden layer

# LSTM architecture
lstm_hidden_sz = 20   # Number of units in the LSTM hidden state
lstm_n_layer = 1      # Number of stacked LSTM layers