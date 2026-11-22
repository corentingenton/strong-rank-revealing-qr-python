function [Q, R, A, B ,C] = partialQR(M,k)
  M1 = M(:,1:k);
  M2 = M(:,k+1:end);
  [Q,Rk] = qr(M1);
  Rc = Q'*M2;
  R = [Rk,Rc];
  B = Rc(1:k,:);
  C = Rc(k+1:end,:);
  A = Rk(1:k,:);
end
