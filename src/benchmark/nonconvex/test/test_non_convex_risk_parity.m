% function test_non_convex_risk_parity(n)
% well now this problem becomes more complicated
% 1/2 xT sigma x + mu x
% s.t.  (1+c)theta - xT Ai x >= 0
%                    xT Ai x >= (1-c)theta
%                       1T x  = 1
clear
n = 100
c = 0.1;
for kk = 1
%     rng(kk);
    P = rand(n,n+100);
    P = P * P' ;
    q = randn(n,1)/n;
    x0 = ones(n,1)/n;
    theta = 0;
    lb = [zeros(n,1);-inf];
    Aeq = ones(1,n);
    beq = 1;
    aub= 1e-9;
    blb= 1e-9;
    [xini,thetaini] = non_convex_risk_parity_c_0(P, q, [], [], x0,theta,Aeq,beq,lb,[]);
    [x2,theta, p, xlist, gaplist] = non_convex_risk_parity(...
        P, q, [], [], xini, thetaini,Aeq,beq,lb,[],c,aub,blb);
    
    
    x_ = ones(n,1)/n;
    f = @(x,n) 1/2 * x(1:n)'* P * x(1:n) + q' * x(1:n);
    options = optimoptions('fmincon','Algorithm','sqp','Display','iter');
    options.MaxFunctionEvaluations  = 100000;
    x0 = [x_;0];
    x = fmincon(@(x) f(x,n),x0,[],[],[Aeq,0],beq,lb,[],...
        @(x) mycon(x,n,P,c),options);
    x = x(1:n);
    %     norm(x2-x)
    f(x,n) -  f(x2,n)
    if f(x,n) -  f(x2,n) < -1
        error('?');
    end
    
    
    
end

residual = [];
for i = 1:size(xlist,2)
    residual = [residual; norm(xlist(:,i) - x)];
    
end

%     f(x,n)
%     f(x2,n)
