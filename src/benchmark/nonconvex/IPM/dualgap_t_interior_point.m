clc;
close all;
P = [4, 1; 1, 4];
q = [3; 4];
x = [-10; -10];
A = [1, 0; 0, 1];
b = [-5; -4];

% Parameters used in Newton's method
r = 1e-5;
alpha = 0.2;
beta = 0.618;

% Parameters used in interior point method
m = size(A, 1);
epsilon = 1e-8;
mulist = [2, 15, 50, 100];
t = 1e-15;

% Plot dual gap
figure; hold on;
xlabel('Number of Newton steps');
ylabel('Logarithm (base 10) of dual gap');
legends = {};

for mu = mulist
   [x, ~, ~, gaplist] = interior_point_qp(...
        P, q, A, b, x, t, r, alpha, beta, mu, epsilon); 
    plot(log10(gaplist));
    legends{end + 1} = sprintf("\\mu = %d", mu);
end
legend(legends);

% Show results
disp('Results: ');
disp('interior method: ');
disp(compute_value_qp(P, q, x));disp(x);
disp('quadprog: ');
[x, p] = quadprog(P, q, A, b);
disp(p);disp(x);