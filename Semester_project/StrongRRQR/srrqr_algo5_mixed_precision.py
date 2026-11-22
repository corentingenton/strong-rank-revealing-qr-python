import numpy as np
from scipy.linalg import qr, solve_triangular
from givens import givens
from householder import *


def srrqr_algo5_mixed_precision(M, f, tol=1e-3, k_stop=1, type_of_algo='tol'):
    # Convert input to float32 for most computations
    M = np.asarray(M, dtype=np.float64)

    # Dimension of the given matrix
    m, n = M.shape

    # Initialization
    k = 0
    R = M.copy()
    P = np.eye(n, dtype=np.float32)

    # Compute the first gamma - use float64 for stability
    gamma = np.linalg.norm(R, axis=0).astype(np.float64)

    # Use float64 for critical accumulation variables
    omega = np.array(R[0, 0], dtype=np.float64)

    # Initialize AB with the correct initial condition
    AB = np.zeros((1, n - 1), dtype=np.float32)

    while True:
        jmax = np.argmax(gamma)
        if type_of_algo == 'tol':
            if k == n - 1:
                return R, P, k
            gamma_max = gamma[jmax]
            if gamma_max < tol:
                print("Gamma max is smaller than the tolerance: stopping the algorithm")
                break
        elif type_of_algo == 'rank':
            if k == k_stop:
                print("k is equal to k_stop: stopping the algorithm")
                break
        print(f"k: {k}")

        # Exchange columns k and k+j_max
        P[:, [k, k + jmax]] = P[:, [k + jmax, k]]
        R[:, [k, k + jmax]] = R[:, [k + jmax, k]]
        gamma[0], gamma[jmax] = gamma[jmax], gamma[0]

        if k >= 1:
            AB[:, [0, jmax]] = AB[:, [jmax, 0]]

        # Householder reflection - convert local slice to float64 for stability
        R_slice = R[k:, k:].astype(np.float64)
        R_slice = householder_reflection_first_column(R_slice)
        R[k:, k:] = R_slice.astype(np.float32)

        ga = R[k, k]
        c_T = R[k, k + 1:]

        if k == 0:
            AB = np.atleast_2d(c_T / ga).astype(np.float32)
            omega = np.array(ga, dtype=np.float64)
        else:
            u, U = AB[:k, 0], AB[:k, 1:]

            # Use float64 for division and outer product to prevent precision loss
            AB_update = np.vstack([
                U.astype(np.float64) - np.outer(u.astype(np.float64), c_T.astype(np.float64)) / ga,
                c_T / ga
            ]).astype(np.float32)
            AB = AB_update

            # Accumulation in float64
            tmp = 1 / omega ** 2 + u.astype(np.float64) ** 2 / ga ** 2
            omega = np.append(np.sqrt(1 / tmp), ga)

        # Update gamma with float64 computations
        gamma_update = np.sqrt(np.maximum(0, gamma[1:].astype(np.float64) ** 2 - c_T.astype(np.float64) ** 2))
        gamma = gamma_update.astype(np.float32)
        gamma = gamma[:-1]

        l = 0
        while True:
            if l > 100:
                print("won't work")

            # Ensure AB and gamma/omega are correctly shaped
            AB_abs = np.abs(AB.astype(np.float64))
            gamma_float64 = gamma.astype(np.float64)
            omega_float64 = omega.astype(np.float64)

            # Carefully construct ratio_matrix to match AB_abs shape
            ratio_matrix = np.outer(gamma_float64[:AB_abs.shape[1]], 1 / omega_float64).T

            max_matrix = np.maximum(AB_abs, ratio_matrix)
            rho_hat = np.max(max_matrix)

            if rho_hat <= f:
                break
            max_idx = np.argmax(max_matrix)
            i, j = np.unravel_index(max_idx, max_matrix.shape)

            P[:, [i, k + j]] = P[:, [k + j, i]]

            # If j > 0 exchange columns k+1 and k+j
            if j > 0:
                AB[:, [0, j]] = AB[:, [j, 0]]
                gamma[[0, j]] = gamma[[j, 0]]
                R[:, [k, k + j]] = R[:, [k + j, k]]

            # Else interchanging the i and k th columns
            if i < k:
                R[:, i:k + 1] = np.concatenate((R[:, i + 1:k + 1], R[:, i:i + 1]), axis=1)
                omega[i:k + 1] = np.concatenate((omega[i + 1:k + 1], omega[i:i + 1]))
                AB[i:k + 1, :] = np.concatenate((AB[i + 1:k + 1, :], AB[i:i + 1, :]), axis=0)

                for ii in range(i, k):
                    # Use float64 for Givens rotation to improve stability
                    G = givens(R[ii, ii].astype(np.float64), R[ii + 1, ii].astype(np.float64))
                    if G[0, :] @ np.array([R[ii, ii], R[ii + 1, ii]]) < 0:
                        G = -G  # guarantee R(ii,ii) > 0
                    R_slice = G @ R[ii:ii + 2, :].astype(np.float64)
                    R[ii:ii + 2, :] = R_slice.astype(np.float32)

                if R[k, k] < 0:
                    R[k, :] = -R[k, :]

            # Third step : zeroing out the below-diag of k+1 th columns
            if k < n - 1:
                for ii in range(k + 1, m):
                    # Use float64 for Givens rotation to improve stability
                    G = givens(R[k, k].astype(np.float64), R[ii, k].astype(np.float64))
                    if G[0, :] @ np.array([R[k, k], R[ii, k]]) < 0:
                        G = -G  # guarantee R(k+1,k+1) > 0
                    R_slice = G @ R[[k, ii], :].astype(np.float64)
                    R[[k, ii], :] = R_slice.astype(np.float32)

            # Get the different values needed for the update
            ga = R[k, k]
            mu = R[k, k + 1] / ga
            if k < n:
                nu = R[k + 1, k + 1] / ga
            else:
                nu = 0

            # Use float64 for these computations
            rho = np.sqrt(mu ** 2 + nu ** 2)
            ga_bar = ga * rho
            b1 = R[:k, k]
            b2 = R[:k, k + 1]
            c1T = R[k, k + 2:]
            if k + 1 > m:
                c2T = np.zeros_like(c1T)
            else:
                c2T = R[k + 1, k + 2:]

            # Use float64 for these computations
            c1T_bar = (mu * c1T.astype(np.float64) + nu * c2T.astype(np.float64)) / rho
            c2T_bar = (nu * c1T.astype(np.float64) - mu * c2T.astype(np.float64)) / rho
            U = AB[:k, 1:]

            # Modify R
            R_tmp = R.copy()
            R_tmp[:k, k + 1] = b1
            R_tmp[:k, k] = b2
            R_tmp[k, k] = ga_bar
            R_tmp[k, k + 1] = ga * mu / rho
            R_tmp[k + 1, k + 1] = ga * nu / rho
            R_tmp[k, k + 2:] = c1T_bar.astype(np.float32)
            R_tmp[k + 1, k + 2:] = c2T_bar.astype(np.float32)
            R = R_tmp

            # Update AB
            # Use float64 for solve_triangular and other critical computations
            u = solve_triangular(R[:k, :k].astype(np.float64), b1.astype(np.float64), lower=False)
            u1 = AB[:k, 0].copy()

            AB[:k, 0] = ((nu ** 2 * u - mu * u1.astype(np.float64)) / rho ** 2).astype(np.float32)
            AB[k, 0] = mu / rho ** 2
            AB[k, 1:] = c1T_bar / ga_bar

            # Use float64 for complex computations
            AB_update = U.astype(np.float64) + (
                    nu * np.outer(u, c2T_bar) - np.outer(u1.astype(np.float64), c1T_bar)
            ) / ga_bar
            AB[:k, 1:] = AB_update.astype(np.float32)

            # Update gamma
            gamma[0] = ga * nu / rho
            # Use float64 for stability
            gamma[1:] = np.sqrt(
                gamma[1:].astype(np.float64) ** 2 +
                (c2T_bar ** 2) -
                (c2T.astype(np.float64) ** 2)
            ).astype(np.float32)

            # Update omega
            u_bar = u1 + mu * u
            omega[k] = ga_bar
            # Use float64 for stability
            omega[:k] = (
                                omega[:k].astype(np.float64) ** (-2) +
                                u_bar ** 2 / (ga_bar ** 2) -
                                u ** 2 / (ga ** 2)
                        ) ** (-1 / 2)

            l += 1
        k += 1
    return R, P, k