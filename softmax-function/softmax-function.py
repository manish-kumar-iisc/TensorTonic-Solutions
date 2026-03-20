import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    
    nume=np.exp(x-np.max(x,axis=-1, keepdims=True))
    deno=np.sum(nume, axis=-1, keepdims=True)
    out=nume/deno
    return out