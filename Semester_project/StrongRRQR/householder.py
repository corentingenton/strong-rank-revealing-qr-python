import numpy as np

def householder_first_column(A):
    """
    Compute Householder reflection that transforms the first column,
    ensuring the diagonal element is positive.
    """
    A = A.copy().astype(float)

    m = A.shape[0]
    x = A[:, 0].copy()

    x_norm = np.linalg.norm(x)

    if x_norm == 0:
        return A

    # Always make the sign negative to ensure positive diagonal
    sign = -1

    u = x.copy()
    u[0] += sign * x_norm  # Using x_norm directly instead of recalculating

    u_norm = np.linalg.norm(u)
    if u_norm == 0:  # Handle the case where u becomes zero
        return A

    u /= u_norm

    A -= 2.0 * np.outer(u, np.dot(u, A))

    # Ensure the diagonal element is exactly positive (handle numerical precision)
    if A[0, 0] < 0:
        A[0, 0] = abs(A[0, 0])

    return A

