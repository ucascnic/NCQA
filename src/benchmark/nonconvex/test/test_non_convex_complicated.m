clc;
close all;
% well now this problem becomes more complicated
% 1/2 xT sigma x + mu x
% s.t.  (1+c)theta - xT Ai x >= 0
%                    xT Ai x >= (1-c)theta
%                       1T x  = 1
c = 0.8;
for kk =1
    
    rng(kk);
    n = 10;
    P = randn(n,n);
    P = eye(n,n) ;
    q = randn(n,1);
    x = ones(n,1)/n;
    lb = zeros(n,1);
    Aeq = ones(1,n);
    beq = 1;
    % solve by fmincon
    
    f = @(x,n) 1/2 * x(1:n)'*P * x(1:n) + q' * x(1:n);
    options = optimoptions('fmincon','Algorithm','sqp','Display','iter');
    options.MaxFunctionEvaluations  = 100000;
    x0 = [x;0];
    x = fmincon(@(x) f(x,n),x0,[],[],[Aeq,0],beq,[],[],...
        @(x) mycon(x,n,P,c),options);
    
    
    % Parameters used in Newton's method
    r = 1e-7;
    alpha = 0.2;
    beta = 0.618;
    
    
    
    
    % Parameters used in interior point method
    %     epsilon = 1e-7;
    %     mu = 10;
    %     t =  100;
    %     [x2, p, xlist, gaplist] = non_convex_qp(...
    %         P, q, [], [], x, t, r, alpha, beta, mu, epsilon,Aeq,beq,lb,ub);
    %
    %
    %
    %     [x, p] = quadprog(P, q, [], [],Aeq,beq,lb,ub);
    %     norm(x2-x)
    %     if norm(x2-x) > 5e-5
    %         error('?');
    %     end
    
end