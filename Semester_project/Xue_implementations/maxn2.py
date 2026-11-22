import numpy as np


def sign2(x):
    """Helper function to return the sign of elements, with special case for zero."""
    return np.where(x > 0, 1, np.where(x < 0, -1, 0))


def maxn(A, B, kl):
    """
    Max norm estimator using Hager's trick.

    Parameters:
    A (numpy.ndarray): Square matrix A.
    B (numpy.ndarray): Matrix B.
    kl (int): Maximum number of iterations.

    Returns:
    gamma (float): Maximum norm estimate.
    i, j (int): Indices of the max norm in the matrix.
    kl (int): Updated number of iterations.
    """

    m, n = B.shape
    # Initialize x with a random vector, normalized by its L1 norm
    x = np.random.rand(n)
    x = x / np.linalg.norm(x, 1)

    # Initialize x as a uniform vector
    x = np.ones(n) / n
    x0 = x.copy()

    for k in range(kl):
        # Solve A\(B * x) -> in Python, we use np.linalg.solve for the system of equations
        y = np.linalg.solve(A, B @ x)

        # xi is the sign of y
        xi = sign2(y)

        # Calculate z = B' * (A'\xi)
        z = B.T @ np.linalg.solve(A.T, xi)

        # Check stopping criterion
        if np.max(np.abs(z)) < (z.T @ x):
            gamma = np.linalg.norm(y, np.inf)
            gamma = np.max(np.abs(np.linalg.solve(A, B @ x)))
            ix = np.argmax(np.abs(x))
            x = np.zeros_like(x)
            x[ix] = 1
            y = np.linalg.solve(A, B @ x)
            iy = np.argmax(np.abs(y))
            return gamma, iy, ix, k

        else:
            j = np.argmax(np.abs(z))
            x = np.zeros(n)
            x[j] = 1

        # If the new x is identical to the previous one, stop
        if np.linalg.norm(x - x0) == 0:
            kl = k
            gamma = np.max(np.abs(np.linalg.solve(A, B @ x)))
            gamma = np.linalg.norm(y, np.inf)
            gamma = np.max(np.abs(np.linalg.solve(A, B @ x)))
            ix = np.argmax(np.abs(x))
            x = np.zeros_like(x)
            x[ix] = 1
            y = np.linalg.solve(A, B @ x)
            iy = np.argmax(np.abs(y))
            return gamma, iy, ix, kl

        x0 = x.copy()

    gamma = np.max(np.abs(np.linalg.solve(A, B @ x)))
    ix = np.argmax(np.abs(x))
    x = np.zeros_like(x)
    x[ix] = 1
    y = np.linalg.solve(A, B @ x)
    iy = np.argmax(np.abs(y))

    return gamma, iy, ix, kl
