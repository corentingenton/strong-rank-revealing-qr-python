import numpy as np


def maxn3(B):
    """
    Finds the maximum absolute value in the matrix B and its corresponding indices.

    Parameters:
    B (numpy.ndarray): Input matrix B.

    Returns:
    gamma (float): Maximum absolute value in the matrix.
    i, j (int): Indices of the maximum absolute value.
    """

    # Get the maximum absolute value of each column and the corresponding row indices
    junk = np.max(np.abs(B), axis=0)
    ic = np.argmax(np.abs(B), axis=0)

    # Find the maximum value among the column maxima
    gamma = np.max(junk)
    j = np.argmax(junk)

    # Get the row index corresponding to the maximum value in column j
    i = ic[j]

    return gamma, i, j

