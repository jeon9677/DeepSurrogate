import numpy as np

def save_np(arr, path):
    with open(path, 'wb') as f:
        np.save(f, arr)
