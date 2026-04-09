import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    vector_matrix=np.zeros((seq_len, d_model))

    r, c = np.ogrid[:seq_len, :d_model]
    vector_matrix=np.zeros((seq_len, d_model))
    vector_matrix[:,0::2]=np.sin(r/base**(c[:,0::2]/d_model))
    odd_indices_as_even = c[:, 1::2] - 1
    
    vector_matrix[:,1::2]=np.cos(r/base**(odd_indices_as_even/d_model))
    return vector_matrix

    