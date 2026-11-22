import numpy as np


def householder_reflection_first_column_modified_corr(A):

    A = A.copy().astype(float)

    m = A.shape[0]
    x = A[:, 0].copy()

    x_norm = np.linalg.norm(x)

    if x_norm == 0:
        return A

    #sign = np.sign(x[0]) if x[0] != 0 else 1
    sign = -1
    u = x.copy()
    u[0] += sign * np.linalg.norm(x)

    u_norm = np.linalg.norm(u)
    #print("u norm = ",u_norm)
    if u_norm < 1e-15:
        return A
    
    u /= u_norm

    A -= 2.0 * np.outer(u, np.dot(u, A))

    return A

