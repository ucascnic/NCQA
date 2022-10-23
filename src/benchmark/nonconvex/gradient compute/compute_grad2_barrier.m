function [grad2] = compute_grad2_barrier(P, q, A, b, x, t,Aeq)
    grad2 = compute_grad2_qp(P, q, x);
    residual = b - A * x;
    r = 1 ./ ( t * residual .^ 2);
    grad2 =  grad2 + A' * sparse(diag(r)) * A;
 
    grad2 = [grad2 Aeq'; Aeq  zeros(size(Aeq,1),size(Aeq,1))];
 
end