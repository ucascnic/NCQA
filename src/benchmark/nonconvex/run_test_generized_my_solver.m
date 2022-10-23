function [timelist,output] = run_test_generized_my_solver(ns,rnglist,freq)
timelist = cell(length(ns),1);
output = cell(length(ns),1);
for i = 1:length(ns)
    n = ns(i);
    rng(rnglist(i));
    ls = [];
    outx = [];
    for kk = 1:freq
        P = rand(n,2*n);
        P = P * P'+ diag(10);
        q = randn(n,1);
        x0 = ones(n,1)/n;
        theta = 0;
        Aeq = ones(1,n);
        beq = 1;
        c = 0.8;
        tic
        [xini,thetaini] = non_convex_risk_parity_c_0(P, q, [], [], x0,theta,Aeq,beq,[],[]);
        [x,theta, ~, ~, ~] = non_convex_risk_parity(...
            P, q, [], [], xini, thetaini,Aeq,beq,[],[],c,0,0);
        tt = toc;
        ls = [ls;tt];
        f = @(x,n) 1/2 * x(1:n)'* P * x(1:n) + q' * x(1:n);
        outx = [outx;f([x;theta],n)];
        
    end
    timelist{i} = ls;
    output{i} = outx;
    
end