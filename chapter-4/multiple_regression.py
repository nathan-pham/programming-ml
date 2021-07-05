import numpy as np
import os

__dirname = os.path.dirname(__file__)

X1, X2, X3, Y = np.loadtxt(os.path.join(__dirname, "./pizza_3_vars.txt"), skiprows=1, unpack=True)

"""
weighted sum of inputs: y = (x1 * y1) + (x2 * y2) + (x3 * y3) + ... + b
multi-dimensional loss functions require matrix manipulation

matrix multiplication golden rule: number of columns of m1 must match number of rows of m2
[ 1 2 3 ]     [ 7    8 ]     [ 58,   64 ]
[ 4 5 6 ]  x  | 9   10 |  =  [ 139, 154 ]
              [ 11  12 ]

calculating matrix[1][1] = (1, 2, 3) • (7, 8, 11) = (1 * 7) + (2 * 8) + (3 * 11) = 58

matrix transpose: swaps matrix dimensions, flips matrix over diagonal
[ 1 2 3 ] = tranpose => [ 1 4 ]
[ 4 5 6 ]               [ 2 5 ]
                        [ 3 6 ]
"""