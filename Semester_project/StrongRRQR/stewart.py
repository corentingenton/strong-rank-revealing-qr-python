import numpy as np
from scipy.linalg import orth
from scipy.linalg import qr


def compute_stewart_matrix(m=1024, n=100):
    # Set the same random seed as MATLAB
    np.random.seed(42)  # Use whatever seed you used in MATLAB
    # Generate orthonormal matrices
    # Transpose the random matrices to match MATLAB's behavior

    U = orth(np.random.rand(n, m).T.copy())
    V = orth(np.random.rand(n, n).T.copy())

    # Create singular values
    t = np.arange(1, n//2 + 1)
    SM_1 = 0.8 ** t
    SM_2 = np.zeros(n//2)

    # Create diagonal matrix of singular values
    SIG = np.diag(np.concatenate([SM_1, SM_2]))

    # Generate final matrix with noise
    # Transpose the random noise matrix as well
    M = U @ SIG @ V + 0.1 * SM_1[-1] * np.random.rand(n, m).T.copy()
    """
    Q, R = qr(np.random.rand(m, n), mode='economic')
    Q *= np.sign(np.diag(R)).reshape(1, -1)

    # Create singular values
    t = np.arange(1, n // 2 + 1)
    SM_1 = 0.8 ** t
    SM_2 = np.zeros(n // 2)

    # Create diagonal matrix of singular values
    SIG = np.diag(np.concatenate([SM_1, SM_2]))

    # Generate final matrix with noise
    M = Q @ SIG @ R + 0.1 * SM_1[-1] * np.random.rand(m, n)
    """
    return M

