import numpy as np
from scipy.linalg import qr, solve_triangular
from givens import givens
from householder import householder_first_column


def srrqr_algo5(M, f, tol):
    """
    Strong Rank Revealing QR with an error threshold tol
        A[:, p] = Q * R = Q [R11, R12;
                               0, R22]
    where R11 and R12 satisfies that matrix (inv(R11) * R12) has entries
    bounded by a pre-specified constant which should be not less than 1.

    Input:
        A, matrix,  target matrix that is approximated.
        f, scalar,  constant that bound the entries of calculated (inv(R11) * R12)
        tol, scalar, error threshold.

    Output:
        A[:, p] = [Q1, Q2] * [R11, R12;
                               0, R22]
               approx Q1 * [R11, R12];
        Only truncated QR decomposition is returned as
            Q = Q1,
            R = [R11, R12];
        where each column of R22 is less than tol.

    Reference:
        Gu, Ming, and Stanley C. Eisenstat. "Efficient algorithms for
        computing a strong rank-revealing QR factorization." SIAM Journal
        on Scientific Computing 17.4 (1996): 848-869.

    Note:
        Algorithm 5 in the above ref. is implemented.
    """

    # Check constant bound f
    if f < 1:
        print('parameter f given is less than 1. Automatically set f = 2')
        f = 2

    # Dimension of the given matrix
    m, n = M.shape

    # Initialization
    k = 0
    R = M.copy()
    # C_k = M.copy()
    P_ = np.eye(n)

    AB = solve_triangular(R[:k, :k], R[:k, k:], lower=False)
    gamma = np.linalg.norm(R[k:, k:], axis=0) if k < R.shape[0] else np.zeros(n - k)
    tmp = solve_triangular(R[:k, :k], np.eye(k), lower=False)
    omega = np.linalg.norm(tmp, axis=1) ** (-1)

    while True:
        if np.max(gamma) < tol:
            break
        jmax = np.argmax(gamma)
        #k = k+1
        #gamma = np.linalg.norm(R[k:, k:], axis=0) if k < R.shape[0] else np.zeros(R.shape[1] - k)
        gamma_max = gamma[jmax]

        P = np.eye(R.shape[1])
        R[:, [k, k + jmax]] = R[:, [k + jmax, k]]
        P[[k, k + jmax]] = P[[k + jmax, k]]
        P_ = P_ @ P

        R[k:, k:] = householder_first_column(R[k:, k:])
        # if R[k, k] < 0:    ---> What to do if R[k, k] is negative ??
        #    R[k, k] = np.abs(if R[k, k])

        # Now we update AB, gamma
        u = AB[:, 0]
        U = AB[:, 1:]
        c_T = R[k, k + 1:]

        # Update AB
        AB_1 = U - (1 / gamma_max) * u @ c_T
        AB_2 = (1 / gamma_max) * c_T
        AB = np.concatenate((AB_1, AB_2), axis=0)

        # Update omega
        # Set the k-th element
        omega[k] = gamma_max
        # Create a mask for elements to be updated
        mask = np.arange(len(omega)) != k  # True for all indices except k
        # Calculate the new values for omega where the mask is True
        tmp_1 = 1 / (omega[mask]) ** 2
        tmp_2 = (u[mask] ** 2) / gamma_max ** 2
        omega[mask] = np.sqrt(1 / (tmp_1 + tmp_2))

        for i in range(len(gamma)):
            gamma[i] = np.sqrt(gamma[i + 1] ** 2 - c_T[i] ** 2)

        ## Updating gamma, omega, AB
        #AB = solve_triangular(R[:k, :k], R[:k, k:], lower=False)
        #gamma = np.linalg.norm(R[k:, k:], axis=0) if k < R.shape[0] else np.zeros(n - k)[:, np.newaxis]
        #tmp = solve_triangular(R[:k, :k], np.eye(k), lower=False)
        #omega = (np.linalg.norm(tmp, axis=1)[np.newaxis, :]) ** (-1)

        while True:
            # Compute rho_hat
            AB_abs = np.abs(AB)
            ratio_matrix = gamma / omega
            max_matrix = np.maximum(AB_abs, ratio_matrix)
            rho_hat = np.max(max_matrix)

            if rho_hat < f:
                break

            [i, j] = np.unravel_index(np.argmax(max_matrix), max_matrix.shape)

            # Interchange the i th and (k+j) th column of target matrix A and
            # update QR decomposition (Q, R, p), AB, gamma, and omega.
            # First step : interchanging the k+1 and k+j th columns
            if j > 1:
                AB[:, [0, j]] = AB[:, [j, 0]]
                gamma[[0, j]] = gamma[[j, 0]]
                R[:, [k, k + j]] = R[:, [k + j, k]]
                P[[k, k + j]] = P[[k + j, k]]

            # Second step : interchanging the i and k th columns
            if i < k:
                P[i:k + 1] = np.roll(P[i:k + 1], -1)
                R[:, i:k + 1] = np.roll(R[:, i:k + 1], -1, axis=1)
                omega[i:k + 1] = np.roll(omega[i:k + 1], -1)
                AB[i:k + 1, :] = np.roll(AB[i:k + 1, :], -1, axis=0)
                # givens rotation for the triangulation of R(1:k, 1:k)
                for ii in range(i, k):
                    G = givens(R[ii, ii], R[ii + 1, ii])
                    # print("Using givens rotation")
                    if G[0, :] @ np.array([R[ii, ii], R[ii + 1, ii]]) < 0:
                        G = -G  # guarantee R(ii,ii) > 0
                    R[ii:ii + 2, :] = G @ R[ii:ii + 2, :]
                    #Q[:, ii:ii + 2] = Q[:, ii:ii + 2] @ G.T
                if R[k, k] < 0:
                    R[k, :] = -R[k, :]
                    #Q[:, k] = -Q[:, k]

            # Third step : zeroing out the below-diag of k+1 th columns
            if k < m:
                for ii in range(k + 2, m):
                    G = givens(R[k + 1, k + 1], R[ii, k + 1])
                    if G[0, :] @ np.array([R[k + 1, k + 1], R[ii, k + 1]]) < 0:
                        G = -G  # guarantee R(k+1,k+1) > 0
                    R[[k + 1, ii], :] = G @ R[[k + 1, ii], :]
                    #Q[:, [k + 1, ii]] = Q[:, [k + 1, ii]] @ G.T

            # Fourth step : interchanging the k and k+1 th columns
            P[[k, k + 1]] = P[[k + 1, k]]
            ga = R[k, k]
            mu = R[k, k + 1] / ga
            if k < m:
                nu = R[k + 1, k + 1] / ga
            else:
                nu = 0
            rho = np.sqrt(mu * mu + nu * nu)
            ga_bar = ga * rho
            b1 = R[:k - 1, k]
            b2 = R[:k - 1, k + 1]
            c1T = R[k, k + 2:]
            if k + 1 > m:
                c2T = np.zeros_like(c1T)
            else:
                c2T = R[k + 1, k + 2:]
            c1T_bar = (mu * c1T + nu * c2T) / rho
            c2T_bar = (nu * c1T - mu * c2T) / rho

            # modify R
            R[:k - 1, k] = b2
            R[:k - 1, k + 1] = b1
            R[k, k] = ga_bar
            R[k, k + 1] = ga * mu / rho
            R[k + 1, k + 1] = ga * nu / rho
            R[k, k + 2:] = c1T_bar
            R[k + 1, k + 2:] = c2T_bar

            # update AB
            u = solve_triangular(R[:k - 1, :k - 1], b1, lower=False)
            u1 = AB[:k - 1, 0]
            AB[:k - 1, 0] = (nu * nu * u - mu * u1) / rho / rho
            AB[k, 0] = mu / rho / rho
            AB[k, 1:] = c1T_bar / ga_bar
            AB[:k - 1, 1:] = AB[:k - 1, 1:] + (nu * u * c2T_bar - u1 * c1T_bar) / ga_bar

            # update gamma
            gamma[0] = ga * nu / rho
            gamma[1:] = np.sqrt(gamma[1:] ** 2 + (c2T_bar ** 2) - (c2T ** 2))

            # update omega
            u_bar = u1 + mu * u
            omega[k] = ga_bar
            omega[:k - 1] = (omega[:k - 1] ** (-2) + u_bar ** 2 / (ga_bar * ga_bar) - u ** 2 / (ga * ga)) ** (-1 / 2)

            # Eliminate new R(k+1, k) by orthogonal transformation
            Gk = np.array([[mu / rho, nu / rho], [nu / rho, -mu / rho]])
            #if k < m:
            #    Q[:, [k, k + 1]] = Q[:, [k, k + 1]] @ Gk.T
        k += 1
    # Only return the truncated version of the strong RRQR decomposition
    #Q = Q[:, :k]
    R = R[:k, :]
    return R, P

    ##################################################################################################################
    """
    # pivoting QR first (generally most time-consuming step)
    Q, R, p = qr(A, pivoting=True)

    # make diagonals of R positive.     ## Not sure why ???
    if R.shape[0] == 1 or R.shape[1] == 1:
        ss = np.sign(R[0, 0])
    else:
        ss = np.sign(np.diag(R))
    R = R * ss[:, np.newaxis]
    Q = Q * ss

    # check a sufficiently large rank k to meet the threshold 'tol'    ## Not sure why ???
    k = None
    if R.shape[0] == 1 or R.shape[1] == 1:
        if R[0, 0] > tol:
            k = 1
    else:
        k = np.where(np.diag(R) > tol)[0][-1] + 1

    # special case : each column has norm less than the threshold.    ## Not sure why ???
    if k is None:
        Q = np.zeros((m, 0))
        R = np.zeros((0, n))
        p = np.arange(n)
        return Q, R, p

    # special case : rank k == n     ## Not sure why ???
    if k == n:
        print('Rank equals the number of columns!')
        return Q, R, p
    
    # The following codes are the major part of the strong rank-revealing
    # algorithm which is based on the above pivoting QR.
    # Name of variables are from the reference paper.

    # Initialization of A^{-1}B ( A refers to R11, B refers to R12)
    AB = solve_triangular(R[:k, :k], R[:k, k:], lower=False)

    # Initialization of gamma, i.e., norm of C's columns (C refers to R22)
    gamma = np.zeros(n - k)
    if k != R.shape[0]:
        gamma = np.sqrt(np.sum(R[k:, k:] ** 2, axis=0))

    # Initialization of omega, i.e., reciprocal of inv(A)'s row norm
    tmp = solve_triangular(R[:k, :k], np.eye(k), lower=False)
    omega = 1 / np.sqrt(np.sum(tmp ** 2, axis=1))

    # KEY STEP in Strong RRQR:
    # "while" loop for interchanging the columns from first k columns and
    # the remaining (n-k) columns.

    # STEP 1: Strong RRQR with the fixed rank k.
    # STEP 2: Check whether a smaller rank k can meet the threshold.

    Rm = R.shape[0]

    while True:
        # STEP 1: Strong RRQR with the fixed rank k.
        while True:
            # identify interchanging columns
            tmp = (1 / omega[:, np.newaxis] * gamma) ** 2 + AB ** 2
            i, j = np.unravel_index(np.argmax(tmp > f * f), tmp.shape)

            # if no entry of tmp greater than f*f, no interchange needed and the
            # present Q, R, p is a strong RRQR.
            if tmp[i, j] <= f * f:
                break

            # Interchange the i th and (k+j) th column of target matrix A and
            # update QR decomposition (Q, R, p), AB, gamma, and omega.
            # First step : interchanging the k+1 and k+j th columns
            if j > 0:
                AB[:, [0, j]] = AB[:, [j, 0]]
                gamma[[0, j]] = gamma[[j, 0]]
                R[:, [k, k + j]] = R[:, [k + j, k]]
                p[[k, k + j]] = p[[k + j, k]]

            # Second step : interchanging the i and k th columns
            if i < k:
                p[i:k + 1] = np.roll(p[i:k + 1], -1)
                R[:, i:k + 1] = np.roll(R[:, i:k + 1], -1, axis=1)
                omega[i:k + 1] = np.roll(omega[i:k + 1], -1)
                AB[i:k + 1, :] = np.roll(AB[i:k + 1, :], -1, axis=0)
                # givens rotation for the triangulation of R(1:k, 1:k)
                for ii in range(i, k):
                    G = givens(R[ii, ii], R[ii + 1, ii])
                    print("Using givens rotation")
                    if G[0, :] @ np.array([R[ii, ii], R[ii + 1, ii]]) < 0:
                        G = -G  # guarantee R(ii,ii) > 0
                    R[ii:ii + 2, :] = G @ R[ii:ii + 2, :]
                    Q[:, ii:ii + 2] = Q[:, ii:ii + 2] @ G.T
                if R[k, k] < 0:
                    R[k, :] = -R[k, :]
                    Q[:, k] = -Q[:, k]

            # Third step : zeroing out the below-diag of k+1 th columns
            if k < Rm:
                for ii in range(k + 2, Rm):
                    G = givens(R[k + 1, k + 1], R[ii, k + 1])
                    if G[0, :] @ np.array([R[k + 1, k + 1], R[ii, k + 1]]) < 0:
                        G = -G  # guarantee R(k+1,k+1) > 0
                    R[[k + 1, ii], :] = G @ R[[k + 1, ii], :]
                    Q[:, [k + 1, ii]] = Q[:, [k + 1, ii]] @ G.T

            # Fourth step : interchanging the k and k+1 th columns
            p[[k, k + 1]] = p[[k + 1, k]]
            ga = R[k, k]
            mu = R[k, k + 1] / ga
            if k < Rm:
                nu = R[k + 1, k + 1] / ga
            else:
                nu = 0
            rho = np.sqrt(mu * mu + nu * nu)
            ga_bar = ga * rho
            b1 = R[:k - 1, k]
            b2 = R[:k - 1, k + 1]
            c1T = R[k, k + 2:]
            if k + 1 > Rm:
                c2T = np.zeros_like(c1T)
            else:
                c2T = R[k + 1, k + 2:]
            c1T_bar = (mu * c1T + nu * c2T) / rho
            c2T_bar = (nu * c1T - mu * c2T) / rho

            # modify R
            R[:k - 1, k] = b2
            R[:k - 1, k + 1] = b1
            R[k, k] = ga_bar
            R[k, k + 1] = ga * mu / rho
            R[k + 1, k + 1] = ga * nu / rho
            R[k, k + 2:] = c1T_bar
            R[k + 1, k + 2:] = c2T_bar

            # update AB
            u = solve_triangular(R[:k - 1, :k - 1], b1, lower=False)
            u1 = AB[:k - 1, 0]
            AB[:k - 1, 0] = (nu * nu * u - mu * u1) / rho / rho
            AB[k, 0] = mu / rho / rho
            AB[k, 1:] = c1T_bar / ga_bar
            AB[:k - 1, 1:] = AB[:k - 1, 1:] + (nu * u * c2T_bar - u1 * c1T_bar) / ga_bar

            # update gamma
            gamma[0] = ga * nu / rho
            gamma[1:] = np.sqrt(gamma[1:] ** 2 + (c2T_bar ** 2) - (c2T ** 2))

            # update omega
            u_bar = u1 + mu * u
            omega[k] = ga_bar
            omega[:k - 1] = (omega[:k - 1] ** (-2) + u_bar ** 2 / (ga_bar * ga_bar) - u ** 2 / (ga * ga)) ** (-1 / 2)

            # Eliminate new R(k+1, k) by orthogonal transformation
            Gk = np.array([[mu / rho, nu / rho], [nu / rho, -mu / rho]])
            if k < Rm:
                Q[:, [k, k + 1]] = Q[:, [k, k + 1]] @ Gk.T

        # STEP 2: Check whether k -> k-1 can meet the threshold.
        tmp, i = min(omega), np.argmin(omega)
        # rank (k-1) doesn't meet the threshold, stop
        if tmp > tol:
            break

        # interchanging the i and k th columns
        if i < k:
            p[i:k + 1] = np.roll(p[i:k + 1], -1)
            R[:, i:k + 1] = np.roll(R[:, i:k + 1], -1, axis=1)
            # givens rotation for the triangulation of R(1:k, 1:k)
            for ii in range(i, k):
                G = givens(R[ii, ii], R[ii + 1, ii])
                if G[0, :] @ np.array([R[ii, ii], R[ii + 1, ii]]) < 0:
                    G = -G
                R[ii:ii + 2, :] = G @ R[ii:ii + 2, :]
                Q[:, ii:ii + 2] = Q[:, ii:ii + 2] @ G.T
            if R[k, k] < 0:
                R[k, :] = -R[k, :]
                Q[:, k] = -Q[:, k]
        k -= 1

        # Update A^{-1}B, omega, gamma (can be improved)
        AB = solve_triangular(R[:k, :k], R[:k, k:], lower=False)  # A^{-1} * B
        gamma = np.sqrt(np.sum(R[k:, k:] ** 2, axis=0))
        tmp = solve_triangular(R[:k, :k], np.eye(k), lower=False)
        omega = 1 / np.sqrt(np.sum(tmp ** 2, axis=1))

    # Only return the truncated version of the strong RRQR decomposition
    Q = Q[:, :k]
    R = R[:k, :]
    return Q, R, p
"""