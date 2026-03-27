import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x,p=np.array(x),np.array(p)
    if not np.allclose(np.sum(p),1):
        raise ValueError
    return np.sum(p*x)
