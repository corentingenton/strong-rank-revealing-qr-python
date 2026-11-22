from helpers import *
from randomized_srrqr import randomized_srrqr


def randomized_qdp(M, rank_l=1, f=2, tol=1e-7, k_stop=None, type_of_algo='tol'):
    # Randomized SRRQR
    Q_tilda, R, P, k, _, _ = randomized_srrqr(M, M_sk=None, mixed=False, f=f, tol=tol, k_stop=k_stop,
                                           type_of_algo=type_of_algo)

    # QR decomposition of (R_k)^T
    Q_bar, L_T = scipy.linalg.qr(R[:k + 1, :].T, pivoting=False)

    # Truncated SVD
    U, D_l, V_T = truncated_svd(L_T.T, rank_l)
    L_l = U @ np.diag(D_l) @ V_T

    # QLP factorization
    m = M.shape[0]
    Q = Q_tilda[:m + 1, :k + 1] @ U
    P = P @ Q_bar @ V_T.T
    M_hat_l = Q @ np.diag(D_l) @ P.T

    return Q, D_l, P, M_hat_l


def low_rank_approximation(M, ranks, matrix_name):
    svd_error = []
    error = []
    for rank in ranks:
        U, D, V_T = truncated_svd(M, rank)
        M_svd_l = U @ np.diag(D) @ V_T
        _, _, _, M_hat_l = randomized_qdp(M, rank_l=rank, f=2, tol=1e-7, k_stop=None,
                                          type_of_algo='tol')
        svd_error.append(np.linalg.norm(M_svd_l - M, 2))
        error.append(np.linalg.norm(M_hat_l - M, 2))

    return np.array(svd_error), np.array(error)



