% function test_non_convex_risk_parity(n)
% well now this problem becomes more complicated
% 1/2 xT sigma x + mu x
% s.t.  (1+c)theta - xT Ai x >= 0
%                    xT Ai x >= (1-c)theta
%                       1T x  = 1
function [optimalx,optimaltheta,iter_val] = test_robust_non_convex_risk_parity(n,P,q,Omega,lambda,w)
global iter_val
c = 0.95;
iter_val = [];

lb = zeros(n,1);
Aeq = ones(1,n);
beq = 1;

[U,S,V] = svd(Omega); %X = U*S*V'.
sqrtOmega =  U*diag(sqrt(diag(S)))*V';
x_ = ones(n,1)/n;
f = @(x,n) 1/2 * x(1:n)'* P * x(1:n) + lambda * ( q' * x(1:n)  - w * sum(sqrtOmega * x(1:n)).^2) ;
options = optimoptions('fmincon','Algorithm','sqp');
options = optimoptions(options,'outputfcn',@outfun);
options.MaxFunctionEvaluations  = 100000;
x0 = [x_;0];
[x,~,~,~] = fmincon(@(x) f(x,n),x0,[],[],[Aeq,0],beq,[],[],...
    @(x) mycon(x,n,P,c),options);
optimalx = x(1:n);
optimaltheta = x(n+1);


 
end

 
