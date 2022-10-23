function [x, p, nsteps,theta,iter] = my_barriar_risk_parity_c_0(P, q, A, b, x, theta,...
    r, alpha, beta,Aeq,beq,lb,ub,iter)
%NEWTON_QP Solve a qradratic programming by newton's method
%   alpha: Armijo criterion for line search
%   beta: descent multiplier for line search
p = 0;
nsteps = 0;
n = size(P,1);
k = 1;
maxiter = 1000;

Id = eye(n+1,n+1);

 
Aeqhat = [Aeq,0];
% scaling the Aeq and beq
tau0 = 1e-2;
zeta = 1e-3;
gamma_ = 1;
Px = P * x;
g = (  diag(x) * P + diag(Px));
grad = [g  -1*ones(n,1)];
 
coeff = norm( 2 * ( grad'  * grad ) ,2 );
Aeqhat = Aeqhat * coeff;
beqhat = beq * coeff;
while k < maxiter
    Px = P * x;
    risk = x.* Px;
    g = (   diag(x) * P     + diag(Px));
    grad = [g  -1*ones(n,1)];
    g = risk - theta;
    tau = tau0; 
    temp = grad'  * grad;
%     temp = diag(diag(temp));
    Q = 2 * ( temp ) +  tau * Id;   
    qk =  2 *  grad' * g   -  Q * [x;theta];
    Qk = Q;
%     AA = [Qk Aeqhat'; Aeqhat 0];
%     bb = [-qk; beqhat];
%     xhatt = AA\bb;
    xhatt = quadprog(Qk,qk,[],[],Aeqhat,beqhat,lb);
    xhat = xhatt(1:n);
    thetahat = xhatt(n+1);
    deltax = xhat - x;
    deltatheta = thetahat - theta;
    if  mod(k,10) == 0
        tau0 = tau0 * 1.1;
    end
    iter = iter+ 1;    
    gamma_ =   gamma_ * (1 -  zeta *  gamma_);
    x = x +  gamma_ * (deltax);
    theta = theta +  gamma_ * (deltatheta);
    norm_residual = norm(deltax)+norm(deltatheta); 
    fprintf("gamma_ = %.8f\t, res = %.8f\n",gamma_,norm_residual);
    k = k + 1;
    has_w_converged =  (norm_residual <= r);
    if  has_w_converged
        
        break
    end
end
end