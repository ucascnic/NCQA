function [x, outputval ,xlist, gaplist] = non_convex_qp(P, q, A, b, x, t, r, alpha, beta, mu, epsion,Aeq,beq,lb,ub)
%INTERIOR_POINT_QP Solve a quadratic programming problem by interior point
% method
outputval = 0;
m = length(lb) + length(ub);
i = 1;
xlist = [];
gaplist = [];
rbase = r;
r0 = 1;
% t = m / epsion + 1;
 while (true)
     r = rbase +  r0*exp(-i); 
 
    [x, ~, nsteps] = my_barriar(P, q, [], [], x, t, r, alpha, beta,Aeq,beq,lb,ub);
 
    xlist = [xlist, x];
    gaplist = [gaplist, m * mu / t * ones(1, nsteps)];
    
    if (m / t^2 < epsion) && r <= rbase + 1e-6
        return;
    end
     t = sqrt(mu) * t;

     i = i + 1; 

         
         
 end
end

