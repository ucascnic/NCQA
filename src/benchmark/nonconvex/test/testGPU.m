clear
% n = 2000;
% rng(0);
% A = randn(n,'double');
% A = A' * A;
% f = @() cond(A);
% tic
% for i = 1:4
% 	f()
% end
% toc
% % 用时0.5472秒
% 
% %% 用GPU进行计算
% rng(0);
% A_gpu = gpuArray(A);
% f_gpu   = @() cond(A_gpu);
% tic
% for i = 1:4
% 	f_gpu()
% end
% toc

%%
rng(0);
n = 1e8;
A_gpu = randn(n,1,'double','gpuArray');
A_gpu2 = randn(n,1,'double','gpuArray'); 
A_gpu+A_gpu; 


