function [timelist,output] = run_test_generized_innter_solver(ns,rnglist,freq,algo)
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
        x_ = ones(n,1)/n;
        f = @(x,n) 1/2 * x(1:n)'* P * x(1:n) + q' * x(1:n);
        options = optimoptions('fmincon','Algorithm',algo);
        options.MaxFunctionEvaluations  = 100000;
        x0 = [x_;0];
        Aeq = ones(1,n);
        beq = 1;
        c = 0.8;
        tic
        x = fmincon(@(x) f(x,n),x0,[],[],[Aeq,0],beq,[],[],...
            @(x) mycon(x,n,P,c),options);
        tt = toc;
        ls = [ls;tt];
        outx = [outx;f(x,n)];
        
    end
    timelist{i} = ls;
    output{i} = outx;
    
end