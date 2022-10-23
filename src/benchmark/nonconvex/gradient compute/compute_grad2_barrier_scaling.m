function [grad2,c] = compute_grad2_barrier_scaling(P, q, A, b, x, t,Aeq)
    grad2 = compute_grad2_qp(P, q, x);
    residual = b - A * x;
    r = 1 ./ ( t * residual .^ 2);
    grad2 =  grad2 + A' * sparse(diag(r)) * A;
    c = norm(grad2,'fro');
 
 
end