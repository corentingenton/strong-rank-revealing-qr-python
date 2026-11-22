from srrqr_algo5 import srrqr_algo5
from helpers import *
import time


def randomized_srrqr_mixed_prec(M, load=False, name='kahan', f=2, tol=1e-7, k_stop=None, type_of_algo='tol'):
    start_time = time.perf_counter()
    if load:
        if name == 'kahan':
            M_sk = np.loadtxt('M_sk_kahan.csv', delimiter=',')
        elif name == 'stewart':
            M_sk = np.loadtxt('M_sk_stewart.csv', delimiter=',')
        elif name == 'devils':
            M_sk = np.loadtxt('M_sk_devils.csv', delimiter=',')
    else:
        M_sk = sketching(M)
    time_sketching = time.perf_counter() - start_time

    M_sk = M_sk.astype(np.float64)
    _, P_sk, k = srrqr_algo5(M_sk, f=f, tol=tol, k_stop=k_stop, type_of_algo=type_of_algo)
    time_algo = time.perf_counter() - start_time

    M = M.astype(np.float32)
    P_sk = P_sk.astype(np.float32)
    Q, R = scipy.linalg.qr(M @ P_sk, pivoting=False)
    time_qr = time.perf_counter() - start_time
    return Q, R, P_sk, k, (time_sketching, time_algo, time_qr)
