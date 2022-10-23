function rosentest(ns,repeat,solver)
lengthns = length(ns);
times = cell(lengthns,repeat);
values = cell(lengthns,repeat);
for i = 1:lengthns
    n = ns(i);
    if n > 3000
        repeat = 10;
    end
    for kk = 1:repeat
        fprintf("test solver %d on %d %d\n",solver,n,kk);
        rng(kk);
        x0 = (rand(n,1)-0.5)*6;
        
        f = @(x,n)  sum( (1-x(1:n-1)).^2 + 100.*(x(2:n)-x(1:n-1).^2).^2 ) ;
        tic;
        switch solver
            case 1
                options = optimoptions('fmincon','Algorithm','sqp',...
                    'MaxIterations',5000,'MaxFunctionEvaluations',5000000);
                x = fmincon(@(x) f(x,n),x0,[],[],[],[],[],[],...
                    [],options);
                time = toc;
            case 2
                options = optimoptions('fmincon','Algorithm','interior-point',...
                    'MaxIterations',5000,'MaxFunctionEvaluations',5000000);
                
                x = fmincon(@(x) f(x,n),x0,[],[],[],[],[],[],...
                    [],options);
                time = toc;
            case 3
                isparallel = 0;
                x = solve_rosenbrock(x0,1e-6,isparallel);
                time = toc;
            case 4
                isparallel = 1;
                x = solve_rosenbrock(x0,1e-6,isparallel);
                
                time = toc;
            otherwise
                error('?')
        end
        times{i,kk} = time;
        values{i,kk} = f(x,n);
    end
end
tt = num2str(floor((now)));
save(['./output/test_rosenbrock_',num2str(solver),'_',tt])