	function [gamma, i, j, kl] = maxn(A,B,kl);
%
%	Max norm estimator, using Hager's trick.
%
	[m,n] = size(B);
	x  = rand(n,1); 
	x = x /norm(x,1);
	x  = ones(n,1)/n;
	x0 = x;
	for k = 1:kl
		y  = A\(B * x);
		xi = sign2(y); 
		z  = B' * (A'\xi);
		if (max(abs(z)) < (z' * x))
			gamma = norm(y,inf);
			gamma = max(abs(A\(B * x)));
			[mx, ix] = max(abs(x));
			x = zeros(length(x), 1);
			x(ix) = 1;
			y = A\(B * x);
			[my, iy] = max(abs(y));
			i = iy; j = ix;
			return
		else
			[zj,j] = max(abs(z));
			x = zeros(n,1); x(j) = 1;
		end
		if (norm(x-x0) == 0)
			kl = k;
			gamma = max(abs(A\(B * x)));
			gamma = norm(y,inf);
			gamma = max(abs(A\(B * x)));
			xx = x;
			[mx, ix] = max(abs(x));
			x = zeros(length(x), 1);
			x(ix) = 1;
			y = A\(B * x);
			[my, iy] = max(abs(y));
			i = iy; j = ix;
			return
		end
		x0 = x;
	end
	gamma = max(abs(A\(B * x)));
	[mx, ix] = max(abs(x));
	x = zeros(length(x), 1);
	x(ix) = 1;
	y = A\(B * x);
	[my, iy] = max(abs(y));
	i = iy; j = ix;
	return
