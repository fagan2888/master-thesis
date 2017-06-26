function [arcoef ser] = uar(y,n_lags)
% AR coefficients and SER .. no constant term
x = lag(y,1);
for i = 2:n_lags
    x = [x lag(y,i)];
end;
tmp = packr([y x]);
y = tmp(:,1);
x = tmp(:,2:end);
bols = inv(x'*x)*(x'*y);
e = y - x*bols;
ssr = e'*e;
ser = sqrt(ssr/(size(x,1)-size(x,2)));
arcoef = bols;

end

