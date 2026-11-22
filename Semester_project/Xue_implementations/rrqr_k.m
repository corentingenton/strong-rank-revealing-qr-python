function [Q, R, P, nb] = rrqr_k(M,k,f)
%
%	computes a rrqr factorization of M with given k
%       M * P = Q* R, where R can be partitioned by [R11 R12;0 R22] 
%       R11's singular values are sufficiently large
%       R22's singular values are sufficiently small
%       norm(M)/norm(R) is bounded by sqrt( 1+f^2*k*(n-k) ), f is a
%       parameter, better be larger than sqrt(2)
	[~ , n] = size(M);
	[~ , R, A, B ,C] = partialQR(M,k);
    P = eye(n);
    invA = A\eye(k);
    nb = 0;
    while 1
        gamma_lst = vecnorm(C,2,1);
        [gamma, jmax] = max(gamma_lst);
        omega_lst = vecnorm(invA,2,2);
        [omega, imax] = max(omega_lst);
        rou = gamma * omega;
        if rou <= f
            [rou, imax, jmax] = maxn3(A\B);
        end
        if rou <= f
            break
        end
        P(:,[jmax+k, imax]) = P(:,[imax, jmax+k]);
        R(:,[jmax+k, imax]) = R(:,[imax, jmax+k]);
        [~ , R, A, B ,C] = partialQR(R,k);
        invA = A\eye(k);
        nb = nb + 1;
    end
    Q = M/R;
    return