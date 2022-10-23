% function test_non_convex_risk_parity(n)
% well now this problem becomes more complicated
% 1/2 xT sigma x + mu x
% s.t.  (1+c)theta - xT Ai x >= 0
%                    xT Ai x >= (1-c)theta
%                       1T x  = 1
clear
n = 100
c = 0.7;
for kk = 1
%      rng(kk);
    P = rand(n,n);
    P = P * P';
    q = randn(n,1);
    x0 = ones(n,1)/n;
    theta = 0;
    lb = [zeros(n,1);-inf];
    Aeq = ones(1,n);
    beq = 1;
    lambda = -0.1;
    w = 1.5;
    Omega = 10*diag(ones(n,1));
    [U,S,V] = svd(Omega); %X = U*S*V'.
    sqrtOmega =  U*diag(sqrt(diag(S)))*V';
    
    
    [xini,thetaini] = non_convex_risk_parity_c_0(P, q, [], [], x0,theta,Aeq,beq,lb,[]);
    [x2,theta, p, iter_val, gaplist] = non_convex_risk_parity_robust(...
        P, q, [], [], xini, thetaini,Aeq,beq,lb,[],c,lambda,w,Omega,0,0);
    
    
    
    x_ = ones(n,1)/n;
    f = @(x,n) 1/2 * x(1:n)'* P * x(1:n) + lambda * ( q' * x(1:n)  - w * sum(sqrtOmega * x(1:n)).^2) ;
    
    options = optimoptions('fmincon','Algorithm','sqp');
    options.MaxFunctionEvaluations  = 100000;
    x0 = [x_;0];
    
    [x,~,flag,~] = fmincon(@(x) f(x,n),x0,[],[],[Aeq,0],beq,lb,[],...
        @(x) mycon(x,n,P,c),options);
    %     norm(x2-x)
    f(x,n) -  f(x2,n)
    if ~flag
        error('?');
    end
 
    
    
    
end

 

%     f(x,n)
%     f(x2,n)
