function [gamma, i,j] = maxn3(B);
%
%
%
[junk,ic] = max(abs(B));
[gamma,j] = max(junk);
i         = ic(j);

