import time
from srrqr_algo5_implementation import srrqr_algo5
from helpers import *
import scipy
import numpy as np


def randomized_srrqr(M, M_sk=None, mixed=False, f=2, tol=1e-7, k_stop=None, type_of_algo='tol'):
    start_time = time.time()

    # Sketching
    if M_sk is None:
        if mixed:
            M_sk = sketching(M, mixed=mixed)
        else:
            M_sk = sketching(M, mixed=mixed)
    end_sketching = time.time()

    # Algo with sketched matrix
    _, P_sk, k, tau_k = srrqr_algo5(M_sk, f=f, tol=tol, k_stop=k_stop, type_of_algo=type_of_algo)
    end_algo = time.time()

    # Perform QR on M @ P
    if mixed:
        M = M.astype(np.float32)
        P_sk = P_sk.astype(np.float32)

    Q, R = scipy.linalg.qr(M @ P_sk, pivoting=False)
    end_qr = time.time()

    # Calculate elapsed times
    time_sketching = end_sketching - start_time
    time_algo = end_algo - end_sketching
    time_qr = end_qr - end_algo

    return Q, R, P_sk, k, [time_sketching, time_algo, time_qr], tau_k
