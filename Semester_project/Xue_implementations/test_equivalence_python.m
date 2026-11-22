clear
clc
seed = 42;  % Set your desired seed
rng(seed);
n = 5;

M = rand(5)';

[Q, R, A, B ,C] = partialQR(M, 3);

%%

[gamma, i, j] = maxn3(M);

%%
clear
clc
rng(42);
M = rand(12, 8);
M2 = rand(12, 8);

%%

seed = 42;  % Set your desired seed
rng(seed);
n = 7;

M = rand(n)';

[Q, R, P, nb] = rrqr_k(M, 3, 1);

%%
disp(M)