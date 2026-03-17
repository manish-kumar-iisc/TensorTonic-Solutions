import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    norm_vec1=np.linalg.norm(x1)
    norm_vec2=np.linalg.norm(x2)
    cosine_dist=np.dot(x1,x2)/(norm_vec1*norm_vec2)
    if label==1:
        dist=1-cosine_dist
    else:
        dist=max(0,cosine_dist-margin)

    return dist