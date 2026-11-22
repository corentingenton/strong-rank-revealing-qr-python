import numpy as np
from scipy.linalg import orth


def compute_devils_stairs_matrix(m=2048, n=100):
    # Set random seed for reproducibility
    np.random.seed(41)  # Use the same seed as your MATLAB code

    # Create singular values vector
    s = np.zeros(n)
    Length = 10
    Nst = n // Length  # integer division

    # Generate the stair pattern
    for i in range(Nst):  # Python uses 0-based indexing
        start_idx = Length * i
        end_idx = Length * (i + 1)
        s[start_idx:end_idx] = -0.6 * i

    # Fill the remaining elements
    s[Length * Nst:] = -0.6 * (Nst - 1)

    # Convert to powers of 10
    s = 10.0 ** s

    # Generate orthogonal matrices and multiply
    # Use transpose to match MATLAB's rand behavior
    M = orth(np.random.rand(n, n).T.copy()) @ np.diag(s) @ orth(np.random.rand(n, n).T.copy())

    # Add zeros to extend to m rows
    M = np.vstack([M, np.zeros((m-n, n))])
    return M
