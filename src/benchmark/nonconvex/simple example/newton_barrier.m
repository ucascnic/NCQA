function [x, p, nsteps] = newton_barrier(P, q, A, b, x, t, r, alpha, beta)
%NEWTON_QP Solve a qradratic programming by newton's method
%   alpha: Armijo criterion for line search
%   beta: descent multiplier for line search
nsteps = 0;
while (true)
    grad2 = compute_grad2_barrier(P, q, A, b, x, t);
    grad = compute_grad_barrier(P, q, A, b, x, t);
    grad2_inv = inv(grad2);
    d = - grad2_inv * grad;
    lambda2 = grad' * grad2_inv * grad;
    if (lambda2 / 2 <= r)
        p = compute_value_barrier(P, q, A, b, x, t);
        return;
    end
    step = backtracking_armijo_barrier(P, q, A, b, x, t, d, alpha, beta);
    x = x + step .* d;
    nsteps = nsteps + 1;
end
end