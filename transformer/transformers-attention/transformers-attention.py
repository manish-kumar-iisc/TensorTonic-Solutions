import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    q_dot_k=torch.matmul(Q,K.transpose(-2,-1))
    d_k=K.shape[-1]
    logits=q_dot_k/math.sqrt(d_k)
    scores=F.softmax(logits, dim=-1)
    attn=torch.matmul(scores,V)
    return attn
