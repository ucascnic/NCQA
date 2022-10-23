function [timelist,output,val] = run_test_robust_innter_solver(ns,rnglist,freq,algo)
timelist = cell(length(ns),1);
output = cell(length(ns),1);
val = cell(length(ns),1);
for i = 1:length(ns)
    n = ns(i);
    rng(rnglist(i));
    ls = [];
    outx = [];valx=[];
    for kk = 1:freq
%         P = rand(n,2*n)./(abs(rand(n,1))*10);

        P = rand(n,2*n);
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
        x_ = ones(n,1)/n;
        f = @(x,n) 1/2 * x(1:n)'* P * x(1:n) + lambda * ( q' * x(1:n)  - w * sum(sqrtOmega * x(1:n)).^2) ;
        
        options = optimoptions('fmincon','Algorithm',algo);
        options.MaxFunctionEvaluations  = 100000;
        x0 = [x_;0];
        
        tic;
        [x,~,~,~] = fmincon(@(x) f(x,n),x0,[],[],[Aeq,0],beq,[zeros(n,1);-inf],[],...
            @(x) mycon(x,n,P,c),options);
        
        tt = toc;
        ls = [ls;tt];
        outx = [outx;x];
        %         f(outx,n)
        valx = [valx;f(outx,n)];
        %          if kk == 2 && i == 1
        %                      x0 = load('matlab','x');
        %           x0 = x0.x;
        %           f(x,n)
        %             1
        %         end
    end
    timelist{i} = ls;
    output{i} = outx;
    val{i} = valx;
end