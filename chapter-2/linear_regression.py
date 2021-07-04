import numpy as np
import os
__dirname = os.path.dirname(__file__)

X, Y = np.loadtxt(os.path.join(__dirname, "./pizza.txt"), skiprows=1, unpack=True)
