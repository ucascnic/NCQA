clear
rng(0);
ns = [  600  800];
 rng(0);
ns = [10 50:50:1000];
nn = length(ns);
rnglist = zeros(nn,1);
freq = 25;

[time_all_g,all_result_my,valrobust] = run_test_robust_my_solver(ns,rnglist,freq);

algo = 'sqp';
[time_all_sqp,all_result_sqp,valspq] = run_test_robust_innter_solver(ns,rnglist,freq,algo);

ns_ipm = [10 50 100 150 200 250 300 400];
algo = 'interior-point';
[time_all_ipm,all_result_ipm,valipm] = run_test_robust_innter_solver(ns_ipm,rnglist,freq,algo);
 
%%
% nn_ipm = length(ns_ipm);
 times_sqp = zeros(nn,1);
% times_ipm = zeros(nn_ipm,1);
% times_g = zeros(nn,1);
for i = 1:nn
    times_sqp(i) = mean(time_all_sqp{i});
%     times_g(i) = mean(time_all_g{i});
end
% for i = 1:nn_ipm
%     times_ipm(i) = mean(time_all_ipm{i});
% end
% figure
% hold on 
% plot(ns,times_sqp);
% plot(ns_ipm,times_ipm);
% plot(ns,times_g);
% legend('sqp','ipm','grad')
tt = int64(now*1000);
save(['output/robust_output',num2str(tt)])

