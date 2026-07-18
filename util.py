# util.py - data preprocessing utilities

import numpy as np

def slide(seq, look_back, hope):
    """
    Create sliding-window input-output pairs from a 1D sequence.

    Args:
        seq: 1D array-like sequence
        look_back: number of past values to use as input
        hope: stride between consecutive windows

    Returns:
        x: 2D array of input windows (samples, look_back)
        y: 1D array of targets (next value after each window)
    """
    x = []
    y = []

    i = 0
    while i < len(seq) - look_back:
        x.append(seq[i:i + look_back])
        y.append(seq[i + look_back])
        i += hope

    x = np.array(x)
    y = np.array(y)

    return x, y