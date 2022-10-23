function [x,theta, outputval ,xlist, gaplist] = ...
    non_convex_risk_parity_c_0(P, q, A, b, x,theta,Aeq,beq,lb,ub)
%INTERIOR_POINT_QP Solve a quadratic programming problem by interior point
% method
% Parameters used in Newton's method
r = 1e-5;
alpha = 0.2;
beta = 0.618;
outputval = 0;
xlist = [];
gaplist = [];
iter = 1;
[x, ~, ~,theta,~] = my_barriar_risk_parity_c_0(P, q, A, b, x, theta, r, alpha, beta,Aeq,beq,lb,ub,iter); 



