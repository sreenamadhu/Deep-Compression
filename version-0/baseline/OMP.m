function x = OMP(D, y, tol, max_iter)
N = size(D, 2);
iter = 0;
res = y;
Supp = [ ];
err = norm(res);
while (err >= tol) && (iter <= max_iter)
iter = iter + 1;
proxy = D'*res;
[val, idx] = max(abs(proxy));
Supp = [ Supp idx ];
x = zeros(N, 1);
x(Supp) = D(:, Supp) \ y; % There are better ways for solving this.
res = y - D*x;
err = norm(res);
end
end