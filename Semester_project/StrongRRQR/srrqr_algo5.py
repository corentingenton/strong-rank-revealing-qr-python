import numpy as np
from scipy.linalg import qr, solve_triangular
from givens import givens
from householder import *


def srrqr_algo5(M, f=2, tol=1e-3, k_stop=None, type_of_algo='tol', sigma_1=1):

    # Dimension of the given matrix
    eps = np.finfo(float).eps
    m, n = M.shape
    number_of_permutations = 0
    number_of_permutations2 = 0
    number_of_norm_used = 0
    # Initialization
    k = 0
    R = np.asarray(M.copy())
    P = np.eye(n)

    # Compute the first gamma
    gamma = np.linalg.norm(R, axis=0)

    while True:
        jmax = np.argmax(gamma)
        if type_of_algo == 'tol':
            if k == n - 1:
                break
            gamma_max = gamma[jmax]
            if gamma_max < tol:
                print("Gamma max is smaller than the tolerance: stopping the algorithm")
                break
        elif type_of_algo == 'rank':
            if k == k_stop:
                print("k is equal to k_stop: stopping the algorithm")
                break

        # Exchange columns k and k+j_max
        P[:, [k, k + jmax]] = P[:, [k + jmax, k]]
        number_of_permutations += 1
        number_of_permutations2 += 1
        R[:, [k, k + jmax]] = R[:, [k + jmax, k]]
        gamma[0], gamma[jmax] = gamma[jmax], gamma[0]

        if k >= 1:
            AB[:, [0, jmax]] = AB[:, [jmax, 0]]

        R[k:, k:] = householder_reflection_first_column2(R[k:, k:])
        ga = R[k, k]
        c_T = R[k, k + 1:]
        if k == 0:
            AB = np.atleast_2d(c_T / ga)
            omega = np.array(ga)
        else:
            u, U = AB[:k, 0], AB[:k, 1:]
            AB = np.vstack([U - np.outer(u, c_T) / ga, c_T / ga])
            tmp = 1 / (omega ** 2) + (u ** 2) / (ga ** 2)
            omega = np.append(np.sqrt(1 / tmp), ga)

        a = gamma[1:] ** 2 - c_T ** 2
        if np.min(a) < 0:
            print(np.min(a))

        # Update gamma
        print("norm\n", np.linalg.norm(R[k:, k:], axis=0))
        print("gamma", gamma)
        gamma[:-1] = np.sqrt(np.maximum(0, gamma[1:] ** 2 - c_T ** 2))
        gamma = gamma[:-1]

        print("gamma", gamma)
        print(AB.shape)
        print(omega.shape)
        print(gamma.shape)

        while True:
            print(k)
            AB_abs = np.abs(AB)
            ratio_matrix = np.outer(gamma, 1/omega).T
            max_matrix = np.maximum(AB_abs, ratio_matrix)
            if k == 50:
                print(np.round(AB_abs, 3))
                print(np.round(ratio_matrix, 3))
                print(np.round(max_matrix, 3))
            rho_hat = np.max(max_matrix)
            print(rho_hat)
            if rho_hat <= f:
                break
            max_idx = np.argmax(max_matrix)
            i, j = np.unravel_index(max_idx, max_matrix.shape)
            print(i, j)
            P[:, [i, k + j]] = P[:, [k + j, i]]
            number_of_permutations += 1

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
                    G = givens(R[ii, ii], R[ii + 1, ii])
                    if G[0, :] @ np.array([R[ii, ii], R[ii + 1, ii]]) < 0:
                        G = -G  # guarantee R(ii,ii) > 0
                    R[ii:ii + 2, :] = G @ R[ii:ii + 2, :]
                if R[k, k] < 0:
                    R[k, :] = -R[k, :]

            # Third step : zeroing out the below-diag of k+1 th columns
            if k < n - 1:
                for ii in range(k + 1, m):
                    G = givens(R[k, k], R[ii, k])
                    if G[0, :] @ np.array([R[k, k], R[ii, k]]) < 0:
                        G = -G  # guarantee R(k+1,k+1) > 0
                    R[[k, ii], :] = G @ R[[k, ii], :]

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
            U = AB[:k, 1:]

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
            AB[:k, 0] = (nu ** 2 * u - mu * u1) / (rho ** 2)
            AB[k, 0] = mu / (rho ** 2)
            AB[k, 1:] = c1T_bar / ga_bar
            AB[:k, 1:] = U + (nu * np.outer(u, c2T_bar) - np.outer(u1, c1T_bar)) / ga_bar

            # Update gamma
            #if np.max(gamma) < np.sqrt(eps) * sigma_1:
            #    gamma = np.linalg.norm(R[k+1:, k+1:], axis=0)
            #    number_of_norm_used += 1
            #else:
            gamma[0] = ga * nu / rho
            gamma[1:] = np.sqrt(np.maximum(0, gamma[1:] ** 2 + (c2T_bar ** 2) - (c2T ** 2)))

            # Update omega
            u_bar = u1 + mu * u
            omega[k] = ga_bar

            # Break down the calculation to identify where it fails
            inner_term = omega[:k] ** (-2)
            ratio_1 = u_bar ** 2 / (ga_bar ** 2)
            ratio_2 = u ** 2 / (ga ** 2)
            sum_term = inner_term + ratio_1 - ratio_2
            # Check for problematic values
            print(f"\nChecking for problematic values in sum_term at iter {k}:")
            print("Negative values?", np.any(sum_term < 0))
            print("Zeros?", np.any(sum_term == 0))
            print("NaN?", np.any(np.isnan(sum_term)))
            print("Inf?", np.any(np.isinf(sum_term)))

            omega[:k] = (omega[:k] ** (-2) + u_bar ** 2 / (ga_bar ** 2) - u ** 2 / (ga ** 2)) ** (-1 / 2)
        k += 1
    return R, P, k, number_of_permutations, number_of_permutations2, number_of_norm_used

    ##################################################################################################################
