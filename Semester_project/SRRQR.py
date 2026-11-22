import numpy as np
from scipy.linalg import qr


def qr_with_column_pivoting(M):
    """QR decomposition with column pivoting using SciPy."""
    Q, R, P = qr(M, mode='economic', pivoting=True)
    return Q, R, P


def compute_residual_norms(A, B, C):
    """Compute norms for the columns of A^-1 B and the columns of C."""
    if A.shape[0] == 0 or A.shape[1] == 0:
        return 0, np.linalg.norm(C, ord=2) if C.size > 0 else 0

    inv_A = np.linalg.inv(A)
    residual_norm_A_B = np.linalg.norm(np.dot(inv_A, B), ord=2) if B.size > 0 else 0
    residual_norm_C = np.linalg.norm(C, ord=2) if C.size > 0 else 0
    return residual_norm_A_B, residual_norm_C


def update_matrix(Q, R, P, jmax, k):
    """Updates the matrix by moving the jmax-th column to the k-th position."""
    R[:, [k, jmax]] = R[:, [jmax, k]]  # Swap columns in R
    P[[k, jmax]] = P[[jmax, k]]  # Swap rows in permutation matrix
    return Q, R, P


def strong_rrqr(M, tol=1e-12, f=10):
    """
    Computes the strong RRQR factorization of the matrix M using Algorithm 5.

    Parameters:
    M (np.ndarray): The input matrix.
    tol (float): Tolerance for checking if a column norm is small enough.
    f (float): Threshold factor to control the strong RRQR condition.

    Returns:
    Q (np.ndarray): Orthogonal matrix Q.
    R (np.ndarray): Upper triangular matrix R.
    P (np.ndarray): Permutation matrix such that M P = Q R.
    """

    m, n = M.shape
    k = 0
    R = M.copy()  # Start with M as the initial R matrix
    P = np.eye(n)  # Identity permutation matrix

    # Step 1: QR decomposition with column pivoting
    Q, R, pivot = qr_with_column_pivoting(M)

    while k < min(m, n):  # Ensure k doesn't exceed matrix dimensions
        # Step 2: Find the column with the largest norm in C
        norms_C = np.linalg.norm(R[k:, k:], axis=0)
        jmax = np.argmax(norms_C)
        if norms_C[jmax] <= tol:
            break

        # Update Q, R, and P matrices by swapping columns
        Q, R, P = update_matrix(Q, R, P, jmax + k, k)

        # Step 3: Compute residual norms ||A^-1 B|| and ||C||
        A = R[:k + 1, :k + 1] if k + 1 <= R.shape[0] and k + 1 <= R.shape[1] else np.zeros((0, 0))
        B = R[:k + 1, k + 1:] if k + 1 <= R.shape[0] else np.zeros((0, 0))
        C = R[k + 1:, k + 1:] if k + 1 < R.shape[0] and k + 1 < R.shape[1] else np.zeros((0, 0))

        residual_norm_A_B, residual_norm_C = compute_residual_norms(A, B, C)

        # Step 4: Check the RRQR conditions and update k
        if residual_norm_A_B > f or residual_norm_C > f:
            # Apply a permutation to move dependent columns
            swap_col = np.argmax(residual_norm_A_B)
            Q, R, P = update_matrix(Q, R, P, swap_col, k)

        k += 1

    return Q, R, P


# Example usage
M = np.random.rand(6, 4)  # Create a random 6x4 matrix
Q, R, P = strong_rrqr(M)

print("Q matrix:")
print(Q)
print("\nR matrix:")
print(R)
print("\nPermutation matrix P:")
print(P)
