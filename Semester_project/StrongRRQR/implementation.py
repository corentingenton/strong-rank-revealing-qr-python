import numpy as np
from scipy.linalg import solve_triangular


def givens(x, y):

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


def exchange_columns_jmax(R, k, jmax, P, gamma, AB=None):
    P[:, [k, k + jmax]] = P[:, [k + jmax, k]]
    R[:, [k, k + jmax]] = R[:, [k + jmax, k]]
    gamma[0], gamma[jmax] = gamma[jmax], gamma[0]

    if k == 0:
        return R, P, gamma
    else:
        AB[:, [0, jmax]] = AB[:, [jmax, 0]]
        return R, P, gamma, AB


def updating_formulas(R, k, gamma, omega=None, AB=None):
    ga = R[k, k]
    c_T = R[k, k + 1:]

    if k == 0:
        # Initialize AB and omega
        AB = np.atleast_2d(c_T / ga)
        omega = np.array(ga)

        # Update gamma
        gamma = np.sqrt(gamma[1:] ** 2 - c_T ** 2)

        return AB, gamma, omega

    else:
        # Update AB and omega
        u, U = AB[:, 0], AB[:, 1:]
        AB = np.vstack([U - np.outer(u, c_T) / ga, c_T / ga])
        tmp = 1 / (omega ** 2) + (u ** 2) / (ga ** 2)
        omega = np.append(np.sqrt(1 / tmp), ga)

        # Update gamma
        gamma = np.sqrt(gamma[1:] ** 2 - c_T ** 2)
        # If we want to make the algorithm work for Kahan
        # C = R[k + 1:, k + 1:]
        # gamma = np.linalg.norm(C, axis=0)

        return AB, gamma, omega


def modifying_formulas(R, k, gamma, omega, AB, m, n):

    # Get the different values needed for the update
    ga = R[k, k]
    mu = R[k, k + 1] / ga
    if k < n:
        nu = R[k + 1, k + 1] / ga
    else:
        nu = 0
    rho = np.sqrt(mu ** 2 + nu ** 2)
    ga_bar = ga * rho
    b1 = R[:k, k]
    b2 = R[:k, k + 1]
    c1T = R[k, k + 2:]
    if k + 1 > m:
        c2T = np.zeros_like(c1T)
    else:
        c2T = R[k + 1, k + 2:]
    c1T_bar = (mu * c1T + nu * c2T) / rho
    c2T_bar = (nu * c1T - mu * c2T) / rho

    # Modify R
    R_tmp = R.copy()
    R_tmp[:k, k + 1] = b1
    R_tmp[:k, k] = b2
    R_tmp[k, k] = ga_bar
    R_tmp[k, k + 1] = ga * mu / rho
    R_tmp[k + 1, k + 1] = ga * nu / rho
    R_tmp[k, k + 2:] = c1T_bar
    R_tmp[k + 1, k + 2:] = c2T_bar
    R = R_tmp

    # Update AB
    u = solve_triangular(R[:k, :k], b1, lower=False)
    u1 = AB[:k, 0].copy()
    AB[:k, 0] = (nu ** 2 * u - mu * u1) / rho ** 2
    AB[k, 0] = mu / rho ** 2
    AB[k, 1:] = c1T_bar / ga_bar
    AB[:k, 1:] = AB[:k, 1:] + (nu * np.outer(u, c2T_bar) - np.outer(u1, c1T_bar)) / ga_bar

    # Update gamma
    gamma[0] = ga * nu / rho
    gamma[1:] = np.sqrt(gamma[1:] ** 2 + (c2T_bar ** 2) - (c2T ** 2))

    # Update omega
    u_bar = u1 + mu * u
    omega[k] = ga_bar
    omega[:k] = (omega[:k] ** (-2) + u_bar ** 2 / (ga_bar ** 2) - u ** 2 / (ga ** 2)) ** (-1 / 2)

    return R, AB, gamma, omega


def reduction_to_special_interchange(R, k, gamma, omega, AB, m, n, i, j):
    # If j > 0 exchange columns k+1 and k+j
    if j > 0:
        AB[:, [0, j]] = AB[:, [j, 0]]
        gamma[[0, j]] = gamma[[j, 0]]
        R[:, [k + 1, k + j + 1]] = R[:, [k + j + 1, k + 1]]

    # Else interchanging the i and k th columns
    if i < k:
        R[:, i:k + 1] = np.concatenate((R[:, i + 1:k + 1], R[:, i:i + 1]), axis=1)
        omega[i:k + 1] = np.concatenate((omega[i + 1:k + 1], omega[i:i + 1]))
        AB[i:k + 1, :] = np.concatenate((AB[i + 1:k + 1, :], AB[i:i + 1, :]), axis=0)

        for ii in range(i, k):
            G = givens(R[ii, ii], R[ii + 1, ii])
            if G[0, :] @ np.array([R[ii, ii], R[ii + 1, ii]]) < 0:
                G = -G  # guarantee R(ii,ii) > 0
            R[ii:ii + 2, :] = G @ R[ii:ii + 2, :]
        if R[k, k] < 0:
            R[k, :] = -R[k, :]

    if k < n - 1:
        for ii in range(k + 2, m):
            G = givens(R[k + 1, k + 1], R[ii, k + 1])
            if G[0, :] @ np.array([R[k + 1, k + 1], R[ii, k + 1]]) < 0:
                G = -G  # guarantee R(k+1,k+1) > 0
            R[[k + 1, ii], :] = G @ R[[k + 1, ii], :]

    return R, AB, gamma, omega


def compute_rho_hat(gamma, omega, AB):
    AB_abs = np.abs(AB)
    ratio_matrix = np.outer(gamma, 1 / omega).T
    max_matrix = np.maximum(AB_abs, ratio_matrix)
    rho_hat = np.max(max_matrix)
    max_idx = np.argmax(max_matrix)
    i, j = np.unravel_index(max_idx, max_matrix.shape)
    return rho_hat, i, j


def householder_first_column(A):

    m = A.shape[0]
    x = A[:, 0].copy()

    x_norm = np.linalg.norm(x)

    if x_norm == 0:
        return A

    # Always make the sign negative to ensure positive diagonal
    sign = -1

    u = x.copy()
    u[0] += sign * x_norm

    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        return A

    u /= u_norm

    A -= 2.0 * np.outer(u, np.dot(u, A))

    return A



