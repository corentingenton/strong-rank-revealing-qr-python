import numpy as np


def kahan(n, theta=1.2):

    s = np.sin(theta)
    c = np.cos(theta)

    A = np.diag([s**i for i in range(n)])
    upper_tri = np.triu(np.ones((n, n)) * -c, k=1)

    # Set the main diagonal to 1
    np.fill_diagonal(upper_tri, 1)

    B =  upper_tri
    K = A @ B

    #diag_elements = np.arange(n, 0, -1)

    # Create a diagonal matrix with these elements
    #matrix = np.diag(diag_elements)
    #pertubation = pert*matrix

    #K = K + pertubation

    return K