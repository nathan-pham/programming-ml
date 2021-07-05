import numpy as np
import struct
import gzip
import os

__dirname = os.path.dirname(__file__)

def load_images(filename):
    filename = os.path.join(__dirname, filename)

    with gzip.open(filename, "rb") as f:
        _ignored, n_images, columns, rows = struct.unpack(">IIII", f.read(16))
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        return all_pixels.reshape(n_images, columns * rows)

def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)

X_TRAIN = prepend_bias(load_images("./train-images-idx3-ubyte.gz"))
X_TEST = prepend_bias(load_images("./t10k-images-idx3-ubyte.gz"))