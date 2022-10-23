function grad = compute_grad_barrier(P, q, A, b, x, t,beq)
    grad = compute_grad_qp(P, q, x);
    residual = (b - A * x)*t;
    grad =   grad + A' * (1 ./ residual);
    grad = [grad;zeros(size(beq))];
    
end