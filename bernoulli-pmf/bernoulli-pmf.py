import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here
    
    pmf=[ p if val==1 else 1-p for val in x ]
    pmf=np.array(pmf)
    return (pmf, p, p*(1-p))