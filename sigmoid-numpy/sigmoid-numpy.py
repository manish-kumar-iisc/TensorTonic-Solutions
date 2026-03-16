import numpy as np
import math

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x_array=np.array(x)
    return 1/(1+ np.exp(-x_array))
    # if x is None:
    #     return x
    # else:
    #     # Write code here
    #     sig_x=[1/(1+ np.exp(-val)) for val in x ]
    #     return sig_x
        