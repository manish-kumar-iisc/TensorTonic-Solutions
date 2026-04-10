import numpy as np
from scipy.special import comb

def get_pdf(n,p,k):
    
    pdf_k=comb(n,k) * (p**k) * (1-p)**(n-k)
    return pdf_k
    
def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    pdf=comb(n,k) * (p**k) * (1-p)**(n-k)
    
    
    cdf=sum([ get_pdf(n,p,i) for i in range(k+1)])
    return (pdf,cdf)