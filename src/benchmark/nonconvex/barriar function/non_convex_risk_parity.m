function [x,theta, outputval ,xlist, gaplist] = ...
    non_convex_risk_parity(P, q, A, b, x,theta,Aeq,beq,lb,ub,c,aub,blb)
%INTERIOR_POINT_QP Solve a quadratic programming problem by interior point
% method
% Parameters used in Newton's method
r = 1e-7;
alpha = 0.2;
beta = 0.9;
epsion = 1e-4;
mu = 10;
t =  0.1;
outputval = 0;
n = size(P,1);
m = 2*n;
i = 1;
xlist = [];
gaplist = [];
rbase = r;
r0 = 100;
iter = 1;
while (true)
    r = rbase +  r0*exp(-i);
    [x, ~, ~,theta,iter] = my_barriar_risk_parity(P, q, A, b, x, theta, t, r, alpha, beta,Aeq,beq,lb,ub,c,iter,aub,blb);
    if isinf(x)
        return
    end
    xlist = [xlist, x];
    if (m / t^2 < epsion) && r <= rbase + 1e-5
        return;
    end
    t = sqrt(mu) * t;
    i = i + 1;
    
    
end
end

