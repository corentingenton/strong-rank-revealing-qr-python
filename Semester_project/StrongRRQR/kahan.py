import numpy as np


# Define a function to create a Kahan Matrix
def compute_kahan_matrix(m=8192, n=400, theta=1.2, pert=True):

    s = np.sin(theta)
    c = np.cos(theta)

    A = np.diag([s**i for i in range(n)])
    upper_tri = np.triu(np.ones((n, n)) * -c, k=1)

    # Set the main diagonal to 1
    np.fill_diagonal(upper_tri, 1)

    B = upper_tri
    K = A @ B

    M = np.zeros((m, n))
    M[:n, :n] = K

    if pert:
        # Adding perturbation
        B = np.zeros((m, n))
        B[:n, :n] = 1e3 * 2 ** (-52) * np.diag(np.arange(n, 0, -1))
        M += B
    return M
