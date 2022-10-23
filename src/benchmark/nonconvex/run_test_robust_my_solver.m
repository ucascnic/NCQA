function [timelist,output,val] = run_test_robust_my_solver(ns,rnglist,freq)
timelist = cell(length(ns),1);
output = cell(length(ns),1);
val = cell(length(ns),1);
for i = 1:length(ns)
    n = ns(i);
    rng(rnglist(i));
    ls = [];
    outx = []; valx=[];
    for kk = 1:freq
 
        x0 = ones(n,1)/n;
%         x0 = abs(randn(n,1)+1);
%         x0 = x0/sum(x0);
        theta0 = 0;
 
             P = rand(n,2*n)./(abs(rand(n,1))*10);
        P = P * P'+ eye(n,n)*10;
        q = randn(n,1);
        Aeq = ones(1,n);
        beq = 1;
        c = 0.2;
        lambda = -1;
        w = 0.5;
        Omega = diag(ones(n,1));
        [U,S,V] = svd(Omega); %X = U*S*V'.
        sqrtOmega =  U*diag(sqrt(diag(S)))*V';
        
        f = @(x,n) 1/2 * x(1:n)'* P * x(1:n) + lambda * ( q' * x(1:n)  - w * sum(sqrtOmega * x(1:n)).^2) ;
        tic
        lb = [zeros(n,1);-inf];
        
        [xini,thetaini] = non_convex_risk_parity_c_0(P, q, [], [], x0,theta0,Aeq,beq,lb,[]);
       
    
        [x,theta, p, iter_val, gaplist] = non_convex_risk_parity_robust(...
            P, q, [], [], xini, thetaini,Aeq,beq,lb,[],c,lambda,w,Omega,0,0);
        
        tt = toc;
        x = [x(:);theta];
        ls = [ls;tt];
        outx = [outx;x];
        valx = [valx;f(x,n)];
        if kk == 2 && i == 1
            save('matlab');
        end
    end
    timelist{i} = ls;
    output{i} = outx;
    val{i} = valx;
end