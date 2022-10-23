function [x, p] = newton_qp(P, q, x, r, alpha, beta)
%NEWTON_QP Solve a qradratic programming by newton's method
%   alpha: Armijo criterion for line search
%   beta: descent multiplier for line search
while (true)
    grad2 = compute_grad2_qp(P, q, x);
    grad = compute_grad_qp(P, q, x);
    d = - inv(grad2) * grad;
    lambda2 = grad' * grad2 * grad;
    if (lambda2 / 2 <= r)
        p = compute_value_qp(P, q, x);
        return;
    end
    t = backtracking_armijo_qp(P, q, x, d, alpha, beta);
    x = x + t .* d;
end
end

