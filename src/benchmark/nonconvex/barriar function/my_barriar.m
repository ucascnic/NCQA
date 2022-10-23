function [x, p, nsteps] = my_barriar(P, q, A, b, x, t,...
    r, alpha, beta,Aeq,beq,lb,ub)
%NEWTON_QP Solve a qradratic programming by newton's method
%   alpha: Armijo criterion for line search
%   beta: descent multiplier for line search
p = 0;
nsteps = 0;
n = size(P,1);
gamma_k = 1;
k = 1;


maxiter = 1000;
grad1 = -1./(t*(x-lb));
grad2 = 1./(t*(ub-x));
norms = norm((grad1.^2 + grad2.^2),2);
tau = 1./(norms+10);
while k < maxiter
    
    grad1 = -1./(t*(x-lb));
    grad2 = 1./(t*(ub-x));
    g1 = -log(x - lb)/t;
    g2 = -log(ub - x)/t;
    
    Q = diag( (grad1.^2 + grad2.^2) * 2 + tau); %     Q2 = 2 * ( At  * A1 ) +  tau * Identity;
    qk = 2 *  (grad1.*g1 + grad2.*g2)  -  Q * x;
    Qk = Q+P;
    qk = qk+q;
    
    AA = [Qk Aeq'; Aeq 0];
    bb = [-qk; beq];
    xhatt = AA\bb;
    xhat = xhatt(1:n);
    deltax = xhat - x;
    
    gamma_ = backtracking_armijo_barrier_2(Qk, qk, A, b, x, t, deltax, alpha, beta,beq,lb,ub,1);
    
    x = x +  gamma_ * (deltax);
    %     gamma_k =   gamma_k * (1 -  zeta *  gamma_k);
    fprintf("gamma_ = %.8f\t, res = %.8f\n",gamma_,norm(deltax));
    k = k + 1;
    has_w_converged =  (norm(deltax) <= r);
    if  has_w_converged
        break
    end
end
end