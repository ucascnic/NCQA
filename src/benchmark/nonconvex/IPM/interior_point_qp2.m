function [x, fval, xlist, gaplist] = interior_point_qp2(P, q, A, b, x, t, r, alpha, beta, mu, epsion,Aeq,beq)
%INTERIOR_POINT_QP Solve a quadratic programming problem by interior point
% method

m = size(A, 1);
i = 1;
xlist = [];
gaplist = [];
rbase = r;
r0 = 1e0;
% t = m / epsion + 1;
 while (true)
%     r = rbase +  r0*exp(-i); 
 
    [x, ~, nsteps] = newton_barrier2(P, q, A, b, x, t, r, alpha, beta,Aeq,beq);
 
 
    
    xlist = [xlist, x];
    gaplist = [gaplist, m * mu / t * ones(1, nsteps)];
    
    if (m / t < epsion) && r <= rbase + 1e-12
        fval = compute_value_barrier(P, q, A, b, x, t);
        return;
    end
     t = mu * t;

     i = i + 1; 

         
         
 end
end

