function [x, fval, xlist, gaplist] = interior_point_qp(P, q, A, b, x, t, r, alpha, beta, mu, epsion)
%INTERIOR_POINT_QP Solve a quadratic programming problem by interior point
% method

m = size(A, 1);
i = 1;
xlist = [];
gaplist = [];
t = m / epsion + 1;
% while (true)
    [x, ~, nsteps] = newton_barrier(P, q, A, b, x, t, r, alpha, beta);
    xlist = [xlist, x];
    gaplist = [gaplist, m * mu / t * ones(1, nsteps)];
    
    if (m / t < epsion)
        fval = compute_value_barrier(P, q, A, b, x, t);
        return;
    end
%     t = mu * t;
%     i = i + 1;
% end
end

