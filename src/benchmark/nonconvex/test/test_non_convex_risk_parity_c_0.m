% function test_non_convex_risk_parity(n)  
% well now this problem becomes more complicated
% 1/2 xT sigma x + mu x
% s.t. theta - xT Ai x >= 0
%                    xT Ai x >= theta
%                       1T x  = 1
% it is equivalent to
%  sum (x sigma x - theta)^2
% s.t.  1T x  = 1
            
clear
n = 100;
c = 0.8;
for kk = 0
    
%     rng(kk);
    
    P = randn(n-20,n);
    P = P'*P;
    q = randn(n,1);
    x = ones(n,1)/n;
    
    theta = 1;
    lb = zeros(n,1);
    Aeq = ones(1,n);
    beq = 1;

    

     x_ = ones(n,1)/n;
     f = @(x,n)  sum( (x(1:n) .* (P * x(1:n)) - x(n+1)).^2 ) ;
    options = optimoptions('fmincon','Algorithm','sqp','Display','iter');
    options.MaxFunctionEvaluations  = 100000;
    x0 = [x_;0];
    x = fmincon(@(x) f(x,n),x0,[],[],[Aeq,0],beq,zeros(n+1,1),[],...
        [],options);  
%     if abs(f(x,n)) > 1e-6
%         error('?');
%     end
%     
    
    x0 = ones(n,1)/n;
    % solve by SCIR
    [x2,theta, outputval ,xlist, gaplist] = non_convex_risk_parity_c_0(P, q, [], [], x0,theta,Aeq,beq,[],[]);
    
    f(x,n) - f([x2;theta],n)
%     if f([x2;theta],n)  > 1e-6
%         error('?');
%     end
    
end

%     f(x,n)
%     f(x2,n)
