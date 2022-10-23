clear
rng(0);
ns = [10 20 50 100];
nn = length(ns);
rnglist = zeros(nn,1);
freq = 3;
algo = 'sqp';
[~,all_result_sqp] = run_test_generized_innter_solver(ns,rnglist,freq,algo);
algo = 'interior-point';
[~,all_result_ipm] = run_test_generized_innter_solver(ns,rnglist,freq,algo);
[~,all_result_my] = run_test_generized_my_solver(ns,rnglist,freq);

%%
error_sqp = zeros(nn,1);
error_ipm = zeros(nn,1);
error_g = zeros(nn,1);

for i = 1:nn
    error_sqp(i) = norm(all_result_sqp{i} - all_result_my{i});
    error_ipm(i) = norm(all_result_ipm{i} - all_result_my{i});
%     error_g(i) = norm(all_result_my{i});
end 
 
figure 
hold on
plot(ns,error_sqp);
plot(ns,error_ipm);
% plot(ns,error_g);
legend('sqp','ipm')
% save(['quality_output',num2str(now)])

