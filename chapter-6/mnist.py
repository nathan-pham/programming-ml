import numpy as np
import struct
import gzip
import os

__dirname = os.path.dirname(__file__)

TRAIN_DATASET = os.path.join(__dirname, "./dataset/train-images-idx3-ubyte.gz")
TEST_DATASET = os.path.join(__dirname, "./dataset/t10k-images-idx3-ubyte.gz")

def load_images(filename):
    with gzip.open(filename, "rb") as f:
        _ignored, n_images, columns, rows = struct.unpack(">IIII", f.read(16))
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        return all_pixels.reshape(n_images, columns * rows)

def load_labels(filename):
    with gzip.open(filename, "rb") as f:
        f.read(8) # skip header bytes
        all_labels = f.read()
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 1)

def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)

def encode_fives(Y):
    return (Y == 5).astype(int)

X_TRAIN = prepend_bias(load_images(TRAIN_DATASET))
Y_TRAIN = encode_fives(load_labels(TRAIN_DATASET))

X_TEST = prepend_bias(load_images(TEST_DATASET))
Y_TEST = encode_fives(load_labels(TEST_DATASET))