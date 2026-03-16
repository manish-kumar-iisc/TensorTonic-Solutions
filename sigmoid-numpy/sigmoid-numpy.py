import numpy as np
import math

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x_array=np.array(x)
    return 1/(1+ np.exp(-x_array))
        