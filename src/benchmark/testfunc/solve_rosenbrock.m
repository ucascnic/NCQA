function [x] = ...
    solve_rosenbrock(x,r,isparallel)

n = length(x);

k = 1;
maxiter = 100000;

Id = sparse(eye(n,n));
tau = 1;
gamma_ = 1;
zeta=1e-3;
zerossparse = sparse(zeros(n,n));

 

 
ff = @(x,n)  sum( (1-x(1:n-1)).^2 + 100.*(x(2:n)-x(1:n-1).^2).^2 ) ;
if isparallel
    row = (1:n-1)';
    col = (1:n-1)';
    col = [col;col];
     
    row = [ row;(2:n)'];
    base = 10*ones(n-1,1);    
     
    
end
while k < maxiter
    
    % compute g and f
    g = 1 - x(1:n-1);
    f = 10*(x(2:n)-x(1:n-1).^2);
    % compute grad
    
    
    
    if isparallel
        value = [-20 * x(1:n-1);base];
        gradf = sparse(row,col,value ,n,n-1);
        gradggradgT = Id;
        gradggradgT(n,n) = 0;

    else
        gradggradgT = zerossparse;
        gradf = sparse(zeros(n,n-1));
        for i = 1:n-1
            gradf(i,i) = -20 * x(i);
            gradf(i+1,i) = 10;
            gradggradgT(i,i)  = 1;
        end
 
    end
    
    temp = gradggradgT + gradf  * gradf';
    %     temp = diag(diag(temp));
    %     scale =  max(eig(temp)) + 1;
    Q = 2 * temp   + tau * Id  ;
    gradgg = -g;
    gradgg(n) = 0;
    qk =  2 * gradf * f + 2* gradgg  -  Q * x;
    
    AA = [Q];
    bb = [-qk];
    xhatt = AA\bb;
    
    xhat = xhatt(1:n);
    
    deltax = xhat - x;
    
    gamma_ =   gamma_ * (1 -  zeta *  gamma_);
    %      gamma_ = gamma0/abs(k+1).^1.01   ;
    
    x = x +  gamma_ * (deltax);
%   ff(x,n)
    norm_residual = norm(deltax);
    %     fprintf("gamma_ = %.8f\t, res = %.8f\n",gamma_,norm_residual);
    k = k + 1;
    has_w_converged =  (norm_residual <= r);
    if  has_w_converged
        
        break
    end
end
end




