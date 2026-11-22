from householder import householder_first_column
from implementation import *


def srrqr_algo5(M, f=2, tol=1e-3, k_stop=None, type_of_algo='tol'):

    # Get dimension of the given matrix
    m, n = M.shape

    # Initialization of the different values
    tau_k = 0  # Number of inner permutation
    k = 0
    R = np.asarray(M.copy())
    P = np.eye(n)

    # Compute the first gamma
    gamma = np.linalg.norm(R, axis=0)

    # outer loop
    while True:
        jmax = np.argmax(gamma)
        gamma_max = gamma[jmax]
        if type_of_algo == 'tol':
            if k == n - 1:
                print("k is equal to n-1: stopping the algorithm")
                break
            if gamma_max < tol:
                print("Gamma max is smaller than the tolerance: stopping the algorithm")
                break
        elif type_of_algo == 'rank':
            if k == k_stop:
                print("k is equal to k_stop: stopping the algorithm")
                break

        # Exchange columns k and k+j_max
        if k == 0:
            R, P, gamma = exchange_columns_jmax(R, k, jmax, P, gamma)
        else:
            R, P, gamma, AB = exchange_columns_jmax(R, k, jmax, P, gamma, AB)

        # Zero out elements below the diagonal of the first column
        R[k:, k:] = householder_first_column(R[k:, k:])

        # Updating formulas
        if k == 0:
            AB, gamma, omega = updating_formulas(R, k, gamma)
        else:
            AB, gamma, omega = updating_formulas(R, k, gamma, omega, AB)

        # inner loop
        while True:
            rho_hat, i, j = compute_rho_hat(gamma, omega, AB)
            if rho_hat <= f:
                break
            P[:, [i, k + j]] = P[:, [k + j, i]]  # For P no need to use the reduction procedure
            tau_k += 1

            # Reduction to the special case: (i, j+k) --> (k, k+1)
            R, AB, gamma, omega = reduction_to_special_interchange(R, k, gamma, omega, AB, m, n, i, j)
            # Modifying formulas
            R, AB, gamma, omega = modifying_formulas(R, k, gamma, omega, AB, m, n)

        k += 1
    print(np.round(R, 2))
    return R, P, k, tau_k
# ----------------------------------------------------------------------------------------------------------------------
