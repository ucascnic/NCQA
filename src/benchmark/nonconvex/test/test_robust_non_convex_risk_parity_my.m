function [optimalx,optimaltheta,iter_val] = test_robust_non_convex_risk_parity_my(n,P,q,Omega,lambda,w)
% global iter_val
c = 0.95;
% iter_val = [];
theta = 0;
lb = zeros(n,1);
Aeq = ones(1,n);
beq = 1;
x0 = ones(n,1)/n;
theta0 = 0;

[xini,thetaini] = non_convex_risk_parity_c_0(P, q, [], [], x0,theta0,Aeq,beq,[],[]);
[optimalx,optimaltheta, p, iter_val, gaplist] = non_convex_risk_parity_robust(...
    P, q, [], [], xini, thetaini,Aeq,beq,[],[],c,lambda,w,Omega);
 


 
end