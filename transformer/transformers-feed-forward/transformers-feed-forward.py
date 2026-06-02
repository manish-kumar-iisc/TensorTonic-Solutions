import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    ff_x=np.matmul(x,W1)+b1
    ff_relu=np.maximum(0,ff_x)
    ff_y=np.matmul(ff_relu, W2) + b2

    return ff_y