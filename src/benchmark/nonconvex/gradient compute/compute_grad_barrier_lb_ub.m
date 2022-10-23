function grad = compute_grad_barrier_lb_ub(P, q, A, b, x, t,beq,lb,ub)
grad = compute_grad_qp(P, q, x);
if ~isempty(b)
    residual = (b - A * x)*t;
    grad =   grad + A' * (1 ./ residual);
else
    grad =   grad + 2*log((x-lb))/t.*(1./(t*(x-lb))) - 2*log((ub-x))/t.*(1./(t*(ub-x)));
    
end
grad = [grad;zeros(size(beq))];

end