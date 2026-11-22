import numpy as np
from partialQR import partialQR
from maxn3 import maxn3

def rrqr_k(M, k, f):
    """
    Computes the RRQR factorization of matrix M with given k.

    M * P = Q * R, where R can be partitioned by [R11 R12; 0 R22]
    and the singular values of R11 are large, while R22's are small.

    Parameters:
    M (numpy.ndarray): The input matrix.
    k (int): The desired rank for RRQR.
    f (float): A parameter, should be larger than sqrt(2) for better results.

    Returns:
    Q (numpy.ndarray): The orthogonal matrix Q.
    R (numpy.ndarray): The upper triangular matrix R.
    P (numpy.ndarray): The permutation matrix P.
    nb (int): The number of swaps performed.
    """

    _, n = M.shape
    _, R, A, B, C = partialQR(M, k)

    P = np.eye(n)
    invA = np.linalg.inv(A)
    nb = 0

    while True:
        # Compute norms for gamma and omega
        gamma_lst = np.linalg.norm(C, axis=0)
        gamma, jmax = np.max(gamma_lst), np.argmax(gamma_lst)

        omega_lst = np.linalg.norm(invA, axis=1)
        omega, imax = np.max(omega_lst), np.argmax(omega_lst)

        rou = gamma * omega

        # Check if rou is less than f
        if rou <= f:
            rou, imax, jmax = maxn3(np.linalg.solve(A, B))

        if rou <= f:
            break

        # Swap the columns of P and R
        P[:, [jmax + k, imax]] = P[:, [imax, jmax + k]]
        R[:, [jmax + k, imax]] = R[:, [imax, jmax + k]]

        # Recompute the QR factorization
        _, R, A, B, C = partialQR(R, k)
        invA = np.linalg.inv(A)
        nb += 1

    Q = np.dot(M, np.linalg.inv(R))

    return Q, R, P, nb
