import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    rows, cols=np.shape(A)
    new_rows, new_cols= cols, rows
    new_A=np.empty((new_rows, new_cols))
    for row, row_list in enumerate(A):
        for col,val in enumerate(row_list):
            new_A[col, row]=val
    return new_A
