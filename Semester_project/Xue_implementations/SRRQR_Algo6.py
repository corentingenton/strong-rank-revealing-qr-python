import numpy as np
from scipy.linalg import qr, solve_triangular, inv

# Mostly an adaptation of a MATLAB implementation of Algorithm 6 :
# https://ch.mathworks.com/matlabcentral/fileexchange/69139-strong-rank-revealing-qr-decomposition
# https://math.berkeley.edu/~mgu/MA273/Strong_RRQR.pdf


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


def srrqr_tol(A, f, tol):
    """
    Strong Rank Revealing QR with an error threshold tol
    Args:
        A (ndarray): Target matrix to be approximated.
        f (float): Constant that bounds the entries of inv(R11) * R12.
        tol (float): Error tolerance for truncating the decomposition.
    Returns:
        Q (ndarray): Orthogonal matrix Q from QR decomposition.
        R (ndarray): Upper triangular matrix from QR decomposition.
        p (ndarray): Column pivot indices.
    """

    # Check the constant bound f
    if f <= 1:
        print('Parameter f is less or equal to 1. Automatically setting f = 2.')
        f = 2

    # Get dimensions of A
    m, n = A.shape

    # Perform pivoted QR decomposition
    Q, R, p = qr(A, pivoting=True, mode='economic')

    # Make diagonals of R positive
    ss = np.sign(np.diag(R)) if min(R.shape) > 1 else np.sign(R[0, 0])
    R = R * ss[:, np.newaxis]
    Q = Q * ss

    # Determine rank k where diag(R) > tol
    diagR = np.diag(R)
    k = np.argmax(diagR <= tol)
    k = n if k == 0 else k

    # Special case: each column norm is less than tol
    if k == 0:
        return np.zeros((m, 0)), np.zeros((0, n)), np.arange(n)

    # Special case: rank k == n
    if k == n:
        print('Rank equals the number of columns!')
        return Q, R, p

    # Initialization of A^{-1}B (A refers to R11, B refers to R12)
    AB = solve_triangular(R[:k, :k], R[:k, k:], lower=False)

    # Initialization of gamma (norm of C's columns)
    gamma = np.linalg.norm(R[k:, k:], axis=0) if k < R.shape[0] else np.zeros(n - k)

    # Initialization of omega (reciprocal of inv(A)'s row norm)
    tmp = solve_triangular(R[:k, :k], np.eye(k), lower=False)
    omega = np.linalg.norm(tmp, axis=1) ** (-1)

    # Main loop
    while True:
        # Step 1: Strong RRQR with fixed rank k
        while True:
            tmp = (1 / omega[:, None] @ gamma[None, :]) ** 2 + AB ** 2 # rho squared
            res = np.where(tmp > f * f)
            if len(res[0]) == 0:
                break
            i, j = res[0][0], res[1][0]

            # Swap columns k+j and k+1, and update matrices
            if j > 0:  # Why j>0 and not j>1
                AB[:, [0, j]] = AB[:, [j, 0]]  # For A^-1B, gamma, R and matrix permutation, just interchange
                gamma[[0, j]] = gamma[[j, 0]]  # the columns k+j and k+1
                R[:, [k, k + j]] = R[:, [k + j, k]]
                p[[k, k + j]] = p[[k + j, k]]

            if i < k:
                # Interchanging the i-th and k-th columns
                # Otherwise we need to apply modifying formula of paragraph 4.1
                # For that, we need to cyclically shifts the last k-i+1 columns
                p[i:k + 1] = np.roll(p[i:k + 1], shift=-1)
                R[:, i:k + 1] = np.roll(R[:, i:k + 1], shift=-1, axis=1)
                omega[i:k + 1] = np.roll(omega[i:k + 1], shift=-1)
                AB[i:k + 1, :] = np.roll(AB[i:k + 1, :], shift=-1, axis=0)

                # Givens rotation for triangulation of R(1:k, 1:k)
                # After the shifts, we need to re-triangularize
                for ii in range(i, k):
                    G = givens(R[ii, ii], R[ii + 1, ii])
                    R[ii:ii + 2, :] = G @ R[ii:ii + 2, :]
                    Q[:, ii:ii + 2] = Q[:, ii:ii + 2] @ G.T

                if R[k, k] < 0:
                    R[k, :] = -R[k, :]
                    Q[:, k] = -Q[:, k]

            # Zeroing out the below-diagonal elements of R(k+1, k+1)
            if k < R.shape[0] - 1:
                for ii in range(k + 2, R.shape[0]):
                    G = givens(R[k + 1, k + 1], R[ii, k + 1])
                    R[[k + 1, ii], :] = G @ R[[k + 1, ii], :]
                    Q[:, [k + 1, ii]] = Q[:, [k + 1, ii]] @ G.T

        # Step 2: Check if we can reduce rank (k -> k-1)
        min_omega, i = np.min(omega), np.argmin(omega)
        if min_omega > tol:
            break

        # Interchanging the i-th and k-th columns
        if i < k:
            p[i:k + 1] = np.roll(p[i:k + 1], shift=-1)
            R[:, i:k + 1] = np.roll(R[:, i:k + 1], shift=-1, axis=1)

            # Givens rotation for triangulation
            for ii in range(i, k):
                G = givens(R[ii, ii], R[ii + 1, ii])
                R[ii:ii + 2, :] = G @ R[ii:ii + 2, :]
                Q[:, ii:ii + 2] = Q[:, ii:ii + 2] @ G.T

            if R[k, k] < 0:
                R[k, :] = -R[k, :]
                Q[:, k] = -Q[:, k]

        k -= 1

        # Update A^{-1}B, omega, and gamma
        AB = solve_triangular(R[:k, :k], R[:k, k:], lower=False)
        gamma = np.linalg.norm(R[k:, k:], axis=0) if k < R.shape[0] else np.zeros(n - k)
        tmp = solve_triangular(R[:k, :k], np.eye(k), lower=False)
        omega = np.linalg.norm(tmp, axis=1) ** (-1)

    # Return truncated version of the strong RRQR decomposition
    return Q[:, :k], R[:k, :], p
