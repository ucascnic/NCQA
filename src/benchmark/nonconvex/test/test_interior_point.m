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
mu = 15;
t = 1e-15;

disp('interior point method: ');
[x2, p, xlist, gaplist] = interior_point_qp(...
    P, q, A, b, x, t, r, alpha, beta, mu, epsilon);

% Plot dual gap
figure;
plot(log10(gaplist));
xlabel('Number of Newton steps');
ylabel('Logarithm (base 10) of dual gap');

% Plot figure of x(t)
figure;
scatter(log(xlist(1, :)), log(xlist(2, :)));
xlabel('logarithm of x_1');
ylabel('logarithm of x_2');
title('Convergence of x(t)');

disp(compute_value_qp(P, q, x));disp(x);
disp('quadprog: ');
[x, p] = quadprog(P, q, A, b);
disp(p);disp(x);