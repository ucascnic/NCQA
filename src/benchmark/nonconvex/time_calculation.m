clear
rng(0);
ns = [10 20 50 300 ];
nn = length(ns);
rnglist = zeros(nn,1);
freq = 3;
algo = 'sqp';
[time_all_sqp,all_result_sqp] = run_test_generized_innter_solver(ns,rnglist,freq,algo);
algo = 'interior-point';
[time_all_ipm,all_result_ipm] = run_test_generized_innter_solver(ns,rnglist,freq,algo);
[time_all_g,all_result_my] = run_test_generized_my_solver(ns,rnglist,freq);


%%
times_sqp = zeros(nn,1);
times_ipm = zeros(nn,1);
times_g = zeros(nn,1);

for i = 1:nn
    times_sqp(i) = mean(time_all_sqp{i});
    times_ipm(i) = mean(time_all_ipm{i});
    times_g(i) = mean(time_all_g{i});
    
end
 
semilogy(ns,times_ipm);hold on;
semilogy(ns,times_sqp);
semilogy(ns,times_g);
legend('sqp','ipm','grad')
save(['output',num2str(now)])

