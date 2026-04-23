import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    tmp1=torch.matmul(Q,K.transpose(-2,-1))
    d_k=K.shape[-1]
    tmp3=tmp1/math.sqrt(d_k)
    scores=F.softmax(tmp3, dim=-1)
    attn=torch.matmul(scores,V)
    return attn
