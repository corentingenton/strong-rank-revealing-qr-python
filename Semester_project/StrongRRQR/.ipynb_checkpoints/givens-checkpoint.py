import numpy as np


def givens(x, y):
    """
    Compute the Givens rotation matrix G that zeroes out the second entry.
    Args:
        x (float): First element.
        y (float): Second element (to be zeroed).
    Returns:
        G (ndarray): Givens rotation matrix of shape (2, 2).
    """

    if y == 0:
        c = 1
        s = 0
    else:
        if abs(y) >= abs(x):
            cotangent = x / y
            s = 1 / np.sqrt(1 + cotangent ** 2)
            c = s * cotangent
        else:
            tangent = y / x
            c = 1 / np.sqrt(1 + tangent ** 2)
            s = c * tangent

    return np.array([[c, s], [-s, c]])
