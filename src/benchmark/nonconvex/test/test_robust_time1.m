ns = [ 200];
nn = length(ns);
rnglist = zeros(nn,1);
freq = 30;
 
[time_all_g,all_result_my] = run_test_robust_my_solver(ns,rnglist,freq);
 