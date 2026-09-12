import numpy as np
txt = open('shakespeare.txt').read()
chars = sorted(set(txt)); stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in txt], dtype=np.int32)
n = int(len(data) * 0.9)
def make_windows(arr, T, n_win, rng):
    idx = rng.integers(0, len(arr) - T, n_win)
    return np.stack([arr[i:i+T] for i in idx])
def get_text_data(T=64, n_train=60000, n_test=2000, seed=0):
    rng = np.random.default_rng(seed)
    return make_windows(data[:n], T, n_train, rng), make_windows(data[n:], T, n_test, rng), len(chars), chars
