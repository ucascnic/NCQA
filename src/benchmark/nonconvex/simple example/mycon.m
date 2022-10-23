function [c,ceq] = mycon(x,n,sigma,c)
x0 = x(1:n);
temp = x0.* (sigma * x0);
c1 =  temp - (1+c)*x(end);
c2 = -temp + (1-c)*x(end);
c = [c1;c2];
ceq = [];