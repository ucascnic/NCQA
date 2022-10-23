function [x, p, nsteps,theta,iter] = my_barriar_risk_parity(P, q, A, b, x, theta, t,...
    r, alpha, beta,Aeq,beq,lb,ub,c,iter,aub,blb)
%NEWTON_QP Solve a qradratic programming by newton's method
%   alpha: Armijo criterion for line search
%   beta: descent multiplier for line search
p = 0;
nsteps = 0;
n = size(P,1);
k = 1;
maxiter = 100;

Id = eye(n+1,n+1);
Phat = blkdiag(P,0);
qhat = [q;0];
Aeqhat0 = [Aeq,0];
% scaling the Aeq and beq

Px = P * x;
risk = x.* Px;
c1_temp = risk - (1+c)*theta - aub;
c2_temp = risk - (1-c)*theta + blb;
c1 = 1./(t*c1_temp);
c2 = 1./(t*c2_temp);
g = (diag(x) * P + diag(Px));
grad1 = diag(c1) * g;
grad2 = diag(c2) * g ;
grad_1 = [grad1  -c1*(1+c)];
grad_2 = [grad2  c2*(c-1)];
grad = [grad_1;grad_2];
gamma_k = 1;zeta = 1e-3;
coeff = norm( 2 * ( grad'  * grad ) , 2);
tau0 = 0.1;
Aeqhat = Aeqhat0 * coeff;
beqhat = beq * coeff;
while k < maxiter
    % xi (sigma) x
    Px = P * x;
    risk = x.* Px;
    c1_temp = risk - (1+c)*theta - aub;
    c2_temp = risk - (1-c)*theta + blb;
    c1 = 1./(t*c1_temp );
    c2 = 1./(t*c2_temp );
    g = ( diag(x) * P + diag(Px));
    grad1 = diag(c1) * g;
    grad2 =  diag(c2) * g ;
    grad = [grad1  -c1*(1+c);grad2  c2*(c-1)];
    g1 = log(-c1_temp)/t;
    g2 = log(c2_temp)/t;
    g = [g1;g2];
    tau = tau0;
    Q = 2 * ( grad'  * grad ) +  tau * Id;
    ss = sum(sum(Q));
    if   any(isnan(ss)) || any(isinf(ss))
        return;
    end
    
    if k<30  
        if cond(Q) > 1e14
            tau = tau * cond(Q);
            Q = Q +  tau * Id;
        end
    end
    gamma_k =   gamma_k * (1 -  zeta *  gamma_k);
    qk =  2 *  grad' * g   -  Q * [x;theta];
    Qk = Q+Phat;
    qk = qk+qhat;
%     AA = [Qk Aeqhat'; Aeqhat 0];
%     bb = [-qk; beqhat];
%     xhatt = AA\bb;
    if ~(isreal(Qk) && isreal(qk) ) 
        return 
    end

    xhatt = quadprog(Qk,qk,[],[],Aeqhat,beqhat,lb);
    xhat = xhatt(1:n);
    thetahat = xhatt(n+1);
    deltax = xhat - x;
    deltatheta = thetahat - theta;
    gamma_ = 1;
    gamma_ = backtracking_armijo_barrier_risk_parity(Qk, qk, A, b, x, theta, t, deltax,deltatheta, alpha, beta,beq,lb,ub,gamma_k,c,P,iter,aub,blb);

    iter = iter+ 1;
 
    x = x +  gamma_ * (deltax);
    theta = theta +  gamma_ * (deltatheta);
    norm_residual = norm(deltax)+norm(deltatheta);
%     fprintf("tau = %.2f gamma_ = %.8f\t, res = %.8f\n",tau,gamma_,norm_residual);
    if gamma_ <= 1e-7  
        return;
    end
    
    k = k + 1;
    has_w_converged =  (norm_residual <= r);
    if  has_w_converged
 
        break
    end
end
end