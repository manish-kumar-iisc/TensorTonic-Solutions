import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    #[[1,2,3,4]]
    #[[2.5, 2.5,2.5,2.5]]
    mean=np.mean(x, axis=-1, keepdims=True)
    
    num=x-mean
    var=np.var(x, axis=-1, keepdims=True)
    print(f"mean:{mean} num:{num} var: {var}" )
    deno_=var + eps
    deno=np.sqrt(deno_)
    temp=num/deno
    layernorm=gamma*temp +beta
    
    return layernorm