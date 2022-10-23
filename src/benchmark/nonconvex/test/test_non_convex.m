clc;
close all;
for kk =1
    
    rng(kk);
    n = 500;
    P = randn(n,n);
    P = P'*P;
    q = randn(n,1);
    x = ones(n,1)/n;
    A = [sparse(eye(n,n)); -sparse(eye(n,n))];
    b = [ones(n,1);ones(n,1)];
    ub = ones(n,1);
    lb = 0*ones(n,1);
    % Parameters used in Newton's method
    r = 1e-7;
    alpha = 0.2;
    beta = 0.618;
    
    % Parameters used in interior point method
    
    epsilon = 1e-7;
    mu = 10;
    t =  100;
    Aeq = ones(1,n);
    beq = 1;
    [x2, p, xlist, gaplist] = non_convex_qp(...
        P, q, [], [], x, t, r, alpha, beta, mu, epsilon,Aeq,beq,lb,ub);
 
    
    
    [x, p] = quadprog(P, q, [], [],Aeq,beq,lb,ub);
    norm(x2-x)
    if norm(x2-x) > 5e-5
        error('?');
    end

end