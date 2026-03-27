import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x,p=np.array(x),np.array(p)
    # prob_sum=np.sum(p)
    # print(prob_sum)
    # print(np.allclose(np.sum(p),1))
    # # if np.allclose(prob_sum,1):
    # tmp=p*x
    # print(tmp)
    if not np.allclose(np.sum(p),1):
        raise ValueError
    return np.sum(p*x)
