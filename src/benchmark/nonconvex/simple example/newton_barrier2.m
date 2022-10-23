function [x, p, nsteps] = newton_barrier2(P, q, A, b, x, t,...
    r, alpha, beta,Aeq,beq)
%NEWTON_QP Solve a qradratic programming by newton's method
%   alpha: Armijo criterion for line search
%   beta: descent multiplier for line search

nsteps = 0;
n = size(P,1);
base_beq = beq;
[~,c] = compute_grad2_barrier_scaling(P, q, A, b, x, t,Aeq);
Aeq = Aeq * c;
beq = beq * c;

while 1
    [grad2] = compute_grad2_barrier(P, q, A, b, x, t,Aeq);
    grad = compute_grad_barrier(P, q, A, b, x, t,beq);
    grad2_inv = inv(grad2);
    d = - grad2_inv * grad;
    lambda2 = - grad' * d;
    if (lambda2 / 2 <= r) ||  (nsteps>20)

        p = compute_value_barrier(P, q, A, b, x, t);
        return;
    end
    step = backtracking_armijo_barrier(P, q, A, b, x, t, d, alpha, beta,base_beq);
    x = x + step .* d(1:n);

    nsteps = nsteps + 1;
end
end